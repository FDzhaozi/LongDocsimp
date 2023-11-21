from methods.ProgDS.discourse_simp import main_disc
from methods.ProgDS.lexical_simp import main_lex
from methods.ProgDS.para_simp import main_para


def main_progds(raw_doc):
    # 参数设置
    shot_set = 2
    if_shot_fixed = False
    if_cot = True
    mode, simple_doc_with_subtitle = main_disc(raw_doc)
    simple_doc_with_simppara = main_para(mode, simple_doc_with_subtitle, shot_set, if_shot_fixed, if_cot)
    simple_doc_with_simplex = main_lex(mode, simple_doc_with_simppara, shot_set=6)

    return simple_doc_with_simplex


if __name__ == '__main__':
    raw_doc = ""
    simple_doc = main_progds(raw_doc)
    simple_doc2 = main_progds(simple_doc)
