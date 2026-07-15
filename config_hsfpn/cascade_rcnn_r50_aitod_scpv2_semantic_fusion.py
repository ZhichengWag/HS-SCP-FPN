# SCPV2 ablation: use segmentation semantic logits for semantic fusion.
#
# Baseline SCPV2 uses the low-frequency semantic embedding as the K/V/gate
# context in SCPV2SemanticFusion.  This ablation keeps the same SCP branch and
# semantic distillation target, but feeds the semantic segmentation logits
# directly into the high-low fusion layer.

_base_ = ['./cascade_rcnn_r50_aitod_scpv2.py']

model = dict(neck=dict(semantic_fusion_source='logits'))

work_dir = '/mnt/e/mmdet5090/work_dirs/cascade_rcnn_r50_aitod_scpv2_semantic_fusion_k150_0.005_epoch24'
