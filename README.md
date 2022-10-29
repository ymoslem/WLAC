# Translation Word-Level Auto-Completion

Code and instructions for our WMT's Word-Level Auto-Completion Shared Task submission paper [Translation Word-Level Auto-Completion: What can we achieve out of the box?](https://arxiv.org/abs/2210.12802)

## Installation

```
pip3 install ctranslate2 nltk tqdm sentencepiece pypinyin jieba
```

## Convertion of OPUS models to CTranslate2

1. Download the model you need from [OPUs models](https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models).
2. Convert the model to the CTranslate2 format:
```
ct2-opus-mt-converter --model_dir opus_model --output_dir ct2_model
```
Optionally, you can use quantization to compress the model and improve efficiency.
```
ct2-opennmt-py-converter --model_path model.pt --quantization int8 --output_dir ct2_model
```
3. Translate using the output CTranslate2 model. Note that SentencePiece tokenization models can be found in the original OPUS model directory.

## Translation with CTranslate2

This is a minimum working example:
```
import ctranslate2
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("source.spm")

source = sp.encode("Hello world!", out_type=str)

translator = ctranslate2.Translator("ende_ctranslate2")
results = translator.translate_batch([source])

output = sp.decode(results[0].hypotheses[0])
print(output)
```

## Auto-Completion with CTranslate2

In this repository, you can find scripts we used for auto-completion. The script [complete.py](complete.py) is for Latin-based target languages (EN-DE, DE-EN, and ZH-EN), while the script [zh-complete.py](zh-complete.py) is for the Chinese target (EN-ZH).

More details about decoding features of in the official [CTranslate2 documentation](https://opennmt.net/CTranslate2/decoding.html).


## Citation

```
@ARTICLE{Moslem2022-WLAC,
  title    = "{Translation Word-Level Auto-Completion: What can we achieve out
              of the box?}",
  author   = "Moslem, Yasmin and Haque, Rejwanul and Way, Andy",
  abstract = "Research on Machine Translation (MT) has achieved important
              breakthroughs in several areas. While there is much more to be
              done in order to build on this success, we believe that the
              language industry needs better ways to take full advantage of
              current achievements. Due to a combination of factors, including
              time, resources, and skills, businesses tend to apply pragmatism
              into their AI workflows. Hence, they concentrate more on
              outcomes, e.g. delivery, shipping, releases, and features, and
              adopt high-level working production solutions, where possible.
              Among the features thought to be helpful for translators are
              sentence-level and word-level translation auto-suggestion and
              auto-completion. Suggesting alternatives can inspire translators
              and limit their need to refer to external resources, which
              hopefully boosts their productivity. This work describes our
              submissions to WMT's shared task on word-level auto-completion,
              for the Chinese-to-English, English-to-Chinese,
              German-to-English, and English-to-German language directions. We
              investigate the possibility of using pre-trained models and
              out-of-the-box features from available libraries. We employ
              random sampling to generate diverse alternatives, which reveals
              good results. Furthermore, we introduce our open-source API,
              based on CTranslate2, to serve translations, auto-suggestions,
              and auto-completions.",
  journal  = "arXiv [cs.CL]",
  month    =  oct,
  year     =  2022,
  url      = "http://arxiv.org/abs/2210.12802"
}
```
