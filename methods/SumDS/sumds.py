from methods.SumDS.gen_paras import seg_doc, simp_para
from methods.SumDS.gen_summary import sum_raw


def main_sumds(raw_doc, shot_set, if_shot_fixed, if_cot):
    # 段落分割
    para_list = seg_doc(raw_doc)

    # 摘要生成
    sum_text = sum_raw(raw_doc)



    # 对每个段落进行简化
    simple_doc = simp_para(sum_text, para_list, shot_set,
                           if_shot_fixed, if_cot)

    # 输出结果
    print(simple_doc)
    return simple_doc


if __name__ == '__main__':
    raw_doc = ""
    # 参数设置
    shot_set = 2
    if_shot_fixed = False
    if_cot = True
    simple_doc = main_sumds(raw_doc, shot_set, if_shot_fixed, if_cot)
