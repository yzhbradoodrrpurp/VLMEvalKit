import os.path as osp
import re
import string
import unicodedata
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge, extract_answer_from_item


def _strip_think_blocks(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()


def _normalise_answer(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _strip_think_blocks(text)
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'answer\s*:', ' ', text)
    text = re.sub(rf'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _numbers(text: Any) -> list[float]:
    if not isinstance(text, str):
        text = str(text)
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', unicodedata.normalize('NFKC', text))]


def _score_blank(prediction: Any, answer: Any) -> tuple[bool, str]:
    pred_nums = _numbers(prediction)
    answer_nums = _numbers(answer)
    if answer_nums:
        hit = bool(pred_nums) and any(np.isclose(pred, ans) for pred in pred_nums for ans in answer_nums)
        return hit, f'numeric: pred={pred_nums}, answer={answer_nums}'

    pred_norm = _normalise_answer(prediction)
    ans_norm = _normalise_answer(answer)
    if not ans_norm:
        return False, 'empty answer'
    if pred_norm == ans_norm:
        return True, f'exact: {pred_norm}'
    if re.search(rf'\b{re.escape(ans_norm)}\b', pred_norm):
        return True, f'substring: {pred_norm}'
    return False, f'normalized mismatch: pred={pred_norm}, answer={ans_norm}'


def _build_blank_judge_prompt(question: Any, prediction: Any, answer: Any) -> str:
    return (
        'Your task is to judge whether the model response expresses the same answer as the ground truth.\n'
        'For numeric answers, accept equivalent numbers and responses that clearly state the same count.\n'
        'Do not require the response wording to exactly match the ground truth.\n\n'
        f'Question: {question}\n'
        f'Ground-truth answer: {answer}\n'
        f'Model response: {prediction}\n\n'
        'If the model response is correct, output Yes. Otherwise, output No.\n'
        'Output only Yes or No.'
    )


def _parse_judge_verdict(text: Any) -> bool | None:
    if not isinstance(text, str):
        text = str(text)
    text = _strip_think_blocks(text).strip().lower()
    match = re.search(r'\b(yes|no)\b', text)
    if match is None:
        return None
    return match.group(1) == 'yes'


class ZoomBenchDataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'ZoomBench_LOCAL': '',
    }

    def load_data(self, dataset):
        return load(osp.join(LMUDataRoot(), f'{dataset}.tsv'))

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line['image_path'])
        question = str(line['question'])
        answer_type = str(line.get('answer_type', line.get('category', ''))).lower()

        if answer_type == 'mcq':
            options = {
                cand: line[cand]
                for cand in 'ABCD'
                if cand in line and not pd.isna(line[cand])
            }
            option_prompt = ''.join(f'{key}. {value}\n' for key, value in options.items())
            question = (
                f'Question: {question}\n'
                f'Options:\n{option_prompt}'
                'Please answer with only the option letter (A, B, C, or D).'
            )
        else:
            question = f'{question}\nPlease answer with the number only.'

        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        data = data.sort_values(by='index').copy()
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        mcq_judge = self._build_mcq_judge(data, judge_kwargs)
        blank_judge = self._build_blank_judge(data, judge_kwargs)

        hits = []
        extracted = []
        logs = []
        for _, row in data.iterrows():
            answer_type = str(row.get('answer_type', row.get('category', ''))).lower()
            if answer_type == 'mcq':
                pred, hit, log = self._score_mcq_with_vlmeval(row, mcq_judge)
                hits.append(float(hit))
                extracted.append(pred)
                logs.append(log)
            else:
                hit, extracted_answer, log = self._score_blank_with_judge(row, blank_judge)
                hits.append(float(hit))
                extracted.append(extracted_answer)
                logs.append(log)

        data['hit'] = hits
        data['extracted_prediction'] = extracted
        data['log'] = logs

        detailed_result_file = get_intermediate_file_path(eval_file, '_zoombench_result')
        dump(data, detailed_result_file)

        acc = self._report_acc(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)
        return acc

    @staticmethod
    def _has_answer_type(data: pd.DataFrame, target: str) -> bool:
        if 'answer_type' in data:
            return any(str(x).lower() == target for x in data['answer_type'])
        return any(str(x).lower() == target for x in data.get('category', []))

    @staticmethod
    def _build_judge(judge_kwargs: dict[str, Any], *, purpose: str):
        model_name = judge_kwargs.get('model', 'exact_matching')
        if model_name in [None, 'exact_matching', 'extract_matching']:
            return None

        kwargs = dict(judge_kwargs)
        kwargs['model'] = model_name
        try:
            judge = build_judge(**kwargs)
            if not judge.working():
                warnings.warn(f'Seed/OpenAI-compatible judge is not working; fallback to ZoomBench {purpose} rules')
                warnings.warn(DEBUG_MESSAGE)
                return None
            return judge
        except Exception as err:
            warnings.warn(f'Failed to build ZoomBench {purpose} judge: {type(err).__name__}: {err}')
            return None

    def _build_mcq_judge(self, data: pd.DataFrame, judge_kwargs: dict[str, Any]):
        if not self._has_answer_type(data, 'mcq'):
            return None
        return self._build_judge(judge_kwargs, purpose='MCQ')

    @staticmethod
    def _build_blank_judge(data: pd.DataFrame, judge_kwargs: dict[str, Any]):
        if 'answer_type' in data:
            has_blank = any(str(x).lower() != 'mcq' for x in data['answer_type'])
        else:
            has_blank = any(str(x).lower() != 'mcq' for x in data.get('category', []))
        if not has_blank:
            return None
        return ZoomBenchDataset._build_judge(judge_kwargs, purpose='open-ended')

    @staticmethod
    def _score_mcq_with_vlmeval(row: pd.Series, judge) -> tuple[str, bool, str]:
        item = row.copy()
        item['GT'] = str(row['answer']).strip()
        result = extract_answer_from_item(judge, item, dataset_name='ZoomBench_LOCAL')
        pred = str(result['opt'])
        hit = pred == item['GT']
        return pred, hit, f'Match Log: {result["log"]}. '

    @staticmethod
    def _score_blank_with_judge(row: pd.Series, judge) -> tuple[bool, str, str]:
        if judge is None:
            hit, log = _score_blank(row['prediction'], row['answer'])
            extracted = str(_numbers(row['prediction']) or _normalise_answer(row['prediction']))
            return hit, extracted, log

        prompt = _build_blank_judge_prompt(row['question'], row['prediction'], row['answer'])
        judge_response = judge.generate(prompt)
        verdict = _parse_judge_verdict(judge_response)
        if verdict is None:
            hit, fallback_log = _score_blank(row['prediction'], row['answer'])
            return hit, str(judge_response), f'judge_unparsed: {judge_response}; fallback: {fallback_log}'
        return verdict, str(judge_response), f'judge: {judge_response}'

    @staticmethod
    def _report_acc(data: pd.DataFrame) -> pd.DataFrame:
        res = defaultdict(list)
        res['split'] = ['none']
        res['Overall'] = [np.mean(data['hit']) * 100]

        if 'answer_type' in data:
            for answer_type in sorted({str(x) for x in data['answer_type']}):
                sub = data[data['answer_type'] == answer_type]
                res[answer_type] = [np.mean(sub['hit']) * 100]

        if 'category' in data:
            for category in sorted({str(x) for x in data['category']}):
                key = f'category:{category}'
                if key in res:
                    continue
                sub = data[data['category'] == category]
                res[key] = [np.mean(sub['hit']) * 100]

        return pd.DataFrame(res).round(2)
