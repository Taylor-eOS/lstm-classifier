from distill_transformer import main_transformer

pred_class, prob_B, logits = main_transformer('infer', 'eval/short.wav')
print(pred_class)
