# 计划

时常: 1.5h

workshop 流程:

1. 介绍 10min 
2. Automatic Speech Recognition (ASR) 技术
3. Text To Speech (TTS) 技术
4. Cutting Edge Technologies



## Introduction

通过一个可运行的现场 Demo 引入: 语音唤起 (例如 小艾同学), 语音指令 (例如要其进行翻译), 语音输出 (AI 将翻译后结果输出)

PPT 讲解刚才整个 demo 底层的运行流程 (VAD -> ASR -> LLM/NLP -> TTS)

介绍整个语音交互的发展历史



## ASR

1. 介绍音频的基础概念 (采样率, 位深) [目的是向非专业学生介绍音频的存储以及获取方式]
2. 声谱图 (介绍语音处理前最重要的产物是什么, 为什么重要)
3. 深入了解声谱图的创建 (介绍以下概念)
   1. 傅里叶变换(FFT)
   2. 梅尔标度/梅尔频谱
   3. MFCC 和 Filterbanks
4. 当前时代主流的模型都有什么, 核心思想是什么 (CTC loss, Transformer/Conformer)
5. colab 展示一段调用例如 Huggingface, 或是 Openai 服务的代码 (教会用户如何使用)



## TTS

1. 介绍 TTS 技术的演变路径 (如果可以的话通过三段音频或是demo来进行对比)
   1. 早期的拼接法
   2. 中期的参数法
   3. 近代的深度学习法
2.  TTS 实现逻辑
   1. 文本前端(预处理)
   2. 声学模型
   3. 声码器
3. 一段调用 TTS 服务的代码, 展示不同参数 (情感参数, 声音长度等) 下得到的结果



## Future

解释一下当前主流技术的一些缺陷 (ASR -> NLP -> TTS) 高延迟, 信息丢失

介绍一下目前主流技术都在研究端到端 (End-To-End) 模型, 即语音输入到语音输出

最后给出一些学习资源 (Huggingface 上不同的模型), 一些经典 paper, 开源框架如何使用 (伪代码或是简易代码)