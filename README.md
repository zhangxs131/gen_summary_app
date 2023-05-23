# gen_summary_app

使用英文原文，进行摘要，然后翻译为中文对pipeline

运行：

```
python app.py
```



模型效果如下：

| method         | Rouge1      | Rouge2      | Rouge-l     | 参数量 |
| -------------- | ----------- | ----------- | ----------- | ------ |
| bart-base      | 52.0333     | 24.2739     | 35.3008     | 139M   |
| Flan-t5-small  | 47.5162     | 20.3709     | 33.1965     | 80M    |
| Flan-t5-base   | 51.0855     | 23.7663     | 36.0833     | 250M   |
| Flan-t5-large  | **53.9433** | **26.5665** | **38.0003** | 780M   |
| mt5-base       | 44.047      | 18.9984     | 31.6903     | 570M   |
| mbart-large-50 | 52.3947     | 25.6287     | 37.0673     | 610M   |
| bart-large-cnn | 54.2034     | 26.4909     | 36.2654     | 406M   |
| Pegasus-large  | 52.5442     | 25.0031     | 36.818      | 570M   |

---









# gen_title_app

中文原文，生成中文title 的摘要任务

1. 运行：

   ```
   python app.py
   ```

   

2. 实验结果：

| method                 | Rouge1     | Rouge2     | Rouge-l    |
| ---------------------- | ---------- | ---------- | ---------- |
| bart(zero-shot)        | 0.1216     | 0.0615     | 0.1086     |
| Pegasus(zero-shot)     | 0.3816     | 0.2278     | 0.3448     |
| pegusus(fine-tune)     | 0.4062     | 0.2611     | 0.3666     |
| Pegasus(da, fine-tune) | **0.4284** | **0.2678** | **0.3953** |









---

# train_summary

使用transfomrers 进行训练摘要模型

1. 运行

   ```sh
   sh script/train_bart_base.sh
   ```

   