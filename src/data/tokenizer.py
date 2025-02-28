from typing import Optional, Union

import spacy_alignments as tokenizations
from transformers import BatchEncoding, BertJapaneseTokenizer
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class BertJapaneseTokenizerFast(BertJapaneseTokenizer):
    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
            text_pair: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
            return_offsets_mapping: bool = False,
            **kwargs: str
        ) -> BatchEncoding:
        v = super().__call__(text=text,text_pair=text_pair,return_offsets_mapping=False,**kwargs)
        if return_offsets_mapping:
            if type(text) is str:
                z = zip([v["input_ids"]], [text], [text_pair] if text_pair else [""])
            else:
                z = zip(v["input_ids"], text, text_pair if text_pair else [""] * len(text))
            w = []
            for a, b, c in z:
                a2b, b2a = tokenizations.get_alignments(self.convert_ids_to_tokens(a), list(b + c))
                x = []
                for i, t in enumerate(a2b):
                    if t == []:
                        s = (0, 0)
                        if a[i] == self.unk_token_id:
                            j = [[-1]] + [t for t in a2b[0:i] if t > []]
                            k = [t for t in a2b[i + 1:] if t > []] + [[len(b + c)]]
                            s = (j[-1][-1] + 1, k[0][0])
                    elif t[-1] < len(b):
                        s = (t[0], t[-1] + 1)
                    else:
                        s = (t[0] - len(b), t[-1] - len(b) + 1)
                    x.append(s)
                w.append(list(x))
            v["offset_mapping"] = w[0] if type(text) is str else w
        return v

