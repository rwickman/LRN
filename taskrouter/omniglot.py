import os

    # print("langs_chars[-1]", langs_chars[-1])

# x = """Gujarati, Sylheti, Arcadian, Tibetan, Old Church Slavonic (Cyrillic), Angelic, Malay (Jawi-Arabic),
# Sanskrit, Cyrillic, Anglo-Saxon Futhorc, Syriac (Estrangelo), Ge’ez, Japanese (katakana), Keble,
# Manipuri, Alphabet of the Magi, Gurmukhi, Korean, Early Aramaic, Atemayar Qelisayer, Tagalog,
# Mkhedruli (Georgian), Inuktitut (Canadian Aboriginal Syllabics), Tengwar, Hebrew, N’Ko, Grantha,
# Latin, Syriac (Serto), Tifinagh, Balinese, Mongolian, ULOG, Futurama, Malayalam, Oriya, Ojibwe
# (Canadian Aboriginal Syllabics), Avesta, Kannada, Bengali, Japanese (hiragana), Armenian, Aurek-
# Besh, Glagolitic, Asomtavruli (Georgian), Greek, Braille, Burmese (Myanmar), Blackfoot (Cana-
# dian Aboriginal Syllabics), Atlantean"""

# langs = x.split(",")[:20]
# for i in range(len(langs)):
#     langs[i] = langs[i].strip()
#     if "(" in langs[i]:
#         langs[i] = langs[i].replace(" (", "_(")

# background = "/media/data/datasets/omniglot/omniglot/images_background"
# evaluation = "/media/data/datasets/omniglot/omniglot/images_evaluation"

# found_langs = []
# found_langs_idxs = []
# for dir_lang in os.listdir(background):
#     for i, lang in enumerate(langs):
#         if lang in dir_lang:
#             print("BACK dir_lang", dir_lang)
#             found_langs_idxs.append(i)
#             found_langs.append(lang)

# for dir_lang in os.listdir(evaluation):
#     for i, lang in enumerate(langs):
#         if lang in dir_lang:
#             print("EVAL dir_lang", dir_lang)
#             found_langs_idxs.append(i)
#             found_langs.append(lang)

# print(len(found_langs), found_langs)
# print(sorted(found_langs_idxs))
# for i in sorted(found_langs_idxs):
#     print(langs[i])