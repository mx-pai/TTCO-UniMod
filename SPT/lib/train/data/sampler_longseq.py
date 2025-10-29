"""
Long-sequence sampler for anti-drift training.
Samples consecutive frames (>=3) instead of just 2 random frames.
"""
import os.path
import random

import torch
import torch.utils.data
from pytorch_pretrained_bert import BertTokenizer

from lib.utils import TensorDict


def no_processing(data):
    return data


class LongSeqTrackingSampler(torch.utils.data.Dataset):
    """
    Long-sequence sampler that draws one template frame followed by a sequence of consecutive search frames.
    The returned sample mimics the structure of VLTrackingSampler so downstream processing remains identical.
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing,
                 seq_length=3, max_seq_len=40, bert_model='bert-base-uncased', bert_path=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - Probabilities for each dataset
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap between template and the first search frame
            num_search_frames - Not used (kept for API parity)
            num_template_frames - Number of template frames to sample
            processing - Processing pipeline (e.g., SPTProcessing)
            seq_length - Number of consecutive search frames (>=1)
        """
        self.datasets = datasets
        self.seq_length = max(1, seq_length)

        if bert_path is not None and os.path.exists(bert_path):
            self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.max_seq_len = max_seq_len  # NLP sequence length

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        valid = False

        while not valid:
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            if not dataset.is_video_sequence():
                # fallback to standard sampler behaviour
                seq_id = 0
                seq_info_dict = dataset.get_sequence_info(seq_id)
                template_frames, template_anno, _ = dataset.get_frames(seq_id, [0], seq_info_dict)
                search_frames, search_anno, meta_obj = dataset.get_frames(seq_id, [0], seq_info_dict)
                mask_shape = search_frames[0].shape[:2]
                zero_mask = torch.zeros(mask_shape)
                template_masks = [zero_mask.clone() for _ in template_anno['bbox']]
                search_masks = [zero_mask.clone()]
                nlp_sentence = seq_info_dict.get('nlp', '')
            else:
                seq_id = random.randint(0, dataset.get_num_sequences() - 1)
                seq_info_dict = dataset.get_sequence_info(seq_id)
                visible = seq_info_dict['visible']
                num_frames = len(visible)

                if num_frames < 2:
                    continue

                seq_len = min(self.seq_length, max(1, num_frames - 1))

                template_ids = self._sample_visible_ids(visible, num_ids=1,
                                                        max_id=max(1, num_frames - seq_len))
                if not template_ids:
                    continue
                template_id = template_ids[0]

                max_forward_gap = num_frames - template_id - seq_len
                if max_forward_gap < 1:
                    continue
                gap = random.randint(1, min(self.max_gap, max_forward_gap))
                search_ids = list(range(template_id + gap, template_id + gap + seq_len))

                template_frames, template_anno, _ = dataset.get_frames(seq_id, [template_id], seq_info_dict)
                search_frames, search_anno, meta_obj = dataset.get_frames(seq_id, search_ids, seq_info_dict)

                h, w, _ = template_frames[0].shape
                zero_mask = torch.zeros((h, w))
                template_masks_raw = template_anno.get('mask', [zero_mask.clone() for _ in template_anno['bbox']])
                if isinstance(template_masks_raw, torch.Tensor):
                    template_masks = [template_masks_raw]
                else:
                    template_masks = list(template_masks_raw)

                search_masks_raw = search_anno.get('mask', None)
                if search_masks_raw is None:
                    search_masks = [zero_mask.clone() for _ in search_frames]
                else:
                    if isinstance(search_masks_raw, torch.Tensor):
                        search_masks = [search_masks_raw]
                    else:
                        search_masks = list(search_masks_raw)

                nlp_sentence = seq_info_dict.get('nlp', '')

            nl_token_ids, nl_token_masks = self._extract_token_from_nlp(nlp_sentence, self.max_seq_len)

            data = TensorDict({
                'template_images': template_frames,
                'template_anno': template_anno['bbox'],
                'template_masks': template_masks,
                'search_images': search_frames,
                'search_anno': search_anno['bbox'],
                'search_masks': search_masks,
                'nl_token_ids': nl_token_ids,
                'nl_token_masks': nl_token_masks,
                'dataset': dataset.get_name(),
                'test_class': meta_obj.get('object_class_name') if meta_obj is not None else None
            })

            data = self.processing(data)
            valid = data.get('valid', True)

        return data

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _extract_token_from_nlp(self, nlp, seq_length):
        """ Tokenize NLP """
        nlp = nlp if isinstance(nlp, str) else ""
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]

        tokens = ["[CLS]"] + nlp_token + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)

        return torch.tensor(input_ids), torch.tensor(input_mask)
