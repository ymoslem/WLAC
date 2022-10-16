# Word-level auto-completion for EN-ZH

import sys
import json
from pprint import pprint
from tqdm import tqdm
import sentencepiece as spm
import ctranslate2
import os
import random
from pypinyin import lazy_pinyin
import jieba
jieba.setLogLevel(20)


file_path = sys.argv[1]
output_path = sys.argv[2]
model_path = sys.argv[3]
sp_source_model = sys.argv[4]
sp_target_model = sys.argv[5]
temp = sys.argv[6]  # 1 or random
detok = sys.argv[7]  # false or ture (for Chinese only)
beam_size = 1


with open(file_path, "r") as jsn:
    json_data = json.load(jsn)
    pprint(json_data[0])

print("File name:", file_path)
print("Number of items:", len(json_data))


# Which GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# For debugging CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def tokenize(text, sp_source_model):
    sp = spm.SentencePieceProcessor(sp_source_model)
    tokens =sp.encode(text, out_type=str)
    tokens = [[">>cmn_Hans<<"] + tk for tk in tokens]
    return tokens


def detokenize(text, sp_target_model):
    sp = spm.SentencePieceProcessor(sp_target_model)
    translation = sp.decode(text)
    return translation


def translate(source_sents, model_path, sp_source_model, sp_target_model, beam_size, st):
    source_sents_tok = tokenize(source_sents, sp_source_model)
    translator = ctranslate2.Translator(model_path, device="cuda")
    translations_tok = translator.translate_batch(source=source_sents_tok,
                                                  beam_size=beam_size,
                                                  batch_type="tokens",
                                                  max_batch_size=4096,
                                                  replace_unknowns=True,
                                                  return_alternatives=True,
                                                  num_hypotheses=10,
                                                  sampling_topk=10,
                                                  sampling_temperature=st,
                                                 )

    translations = [detokenize(hypothesis, sp_target_model) for hypothesis in translations_tok[0].hypotheses]
    return translations

def translate_with_prefix(source_sents, prefix_phrases, model_path, sp_source_model, sp_target_model, beam_size, st):
    source_sents_tok = tokenize(source_sents, sp_source_model)
    prefix_phrases_tok = tokenize(prefix_phrases, sp_target_model)
    translator = ctranslate2.Translator(model_path, "cuda")
    translations_tok = translator.translate_batch(source=source_sents_tok,
                                                  target_prefix=prefix_phrases_tok,
                                                  batch_type="tokens",
                                                  max_batch_size=4096,
                                                  replace_unknowns=True,
                                                  return_alternatives=True,
                                                  beam_size=beam_size,
                                                  num_hypotheses=10,
                                                  sampling_topk=10,
                                                  sampling_temperature=st,
                                                 )
    translations = [detokenize(hypothesis, sp_target_model) for hypothesis in translations_tok[0].hypotheses]
    return translations



num_found = 0

with open(output_path, "w+", encoding='utf-8') as output:
    for item in tqdm(json_data, total=len(json_data)):
        if detok == "true":
            src = item["src"].strip()
            src = detokenize(src.split(), sp_source_model).strip()
        elif detok == "false":
            src = item["src"].strip()
        else:
            print(detok)
        #print(detok)
        #print(src)
        context = item["context_type"]
        prefix = item["left_context"]
        typed = item["typed_seq"]

        found = False
        for i in range(5):
            if found == True:
                break
            else:
                st = random.uniform(1.0, 1.3) if temp == "random" else 1
                # print(st)
                translations = translate([src], model_path, sp_source_model, sp_target_model, beam_size, st)

                # idx = 0
                for translation in translations:
                    # idx += 1

                    if found == True:
                        break
                    else:
                        zh_words = list(jieba.cut(translation.strip()))
                        translation_tok = " ".join(zh_words)
                        words = "".join(lazy_pinyin(translation_tok)).split()
                        for idx, word in enumerate(words):
                            word = word.strip()
                            if word.strip().startswith(typed):
                                # print(idx, "• Typed:", typed, "• Word found:", word)
                                # print(zh_words, words, sep="\n");
                                # print(lazy_pinyin(translation_tok))
                                pinyin = word.strip();
                                # print(pinyin)
                                target = list(zh_words)[idx].strip()
                                # print(target)
                                num_found += 1
                                found = True
                                item["target"] = target
                                break
                            else:
                                item["target"] = ""

        output.write(str(json.dumps(item, indent=4, ensure_ascii=False)))
        output.write(",\n")

print("\nFound:", num_found)
