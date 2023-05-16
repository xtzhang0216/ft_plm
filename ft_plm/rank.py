# 从/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design_df.csv中提取plddt,并按plddt升序排列，把排序结果输出到新文件里
import pandas as pd
# original_design_df = pd.read_csv('/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design_df.csv')
# original_design_df = original_design_df.sort_values(by='plddt', ascending=True)
# original_design_df.to_csv('/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design_df_sorted.csv')
# 从original_design_df_sorted.csv中提取plddt<70的name，根据name分别去/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/all_design_df.csv、original_design_df.csv、partial_design_df.csv、plddt_design_df.csv中提取name对应的plddt和log
all_design_df = pd.read_csv('/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/all_design_df.csv')
original_design_df = pd.read_csv('/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design_df.csv')
partial_design_df = pd.read_csv('/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/partial_design_df.csv')
plddt_design_df = pd.read_csv('/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/plddt_design_df.csv')
original_design_df_sorted = pd.read_csv('/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design_df_sorted.csv')
original_design_df_sorted = original_design_df_sorted[original_design_df_sorted['plddt']<70]
original_design_df_sorted = original_design_df_sorted.reset_index(drop=True)
# 将







