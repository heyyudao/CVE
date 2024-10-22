恶意干扰模型行训练
近期字节出现大瓜，实习生恶意干扰模型训练，造成不少损失，在深挖之后，发现一些可以学习的知识点。并加以分析和分享。

![image](https://github.com/user-attachments/assets/9b1ad881-db34-4f9b-b715-6278d3976d60)
![image](https://github.com/user-attachments/assets/8a216883-7dd7-463f-8659-8b6d62c370cd)

https://www.zaobao.com.sg/news/china/story20241020-5192572

https://www.theguardian.com/technology/2024/oct/21/tiktok-owner-bytedance-sacks-intern-for-allegedly-sabotaging-ai-project

首先尝试在网上收集更多情报 :
在Google搜索更多相关的信息中，找到一个github (田XX 恶意攻击AI集群事件的证据揭露)
https://github.com/JusticeFighterDance/JusticeFighter110

从论坛的讨论中，可以得到相关的信息。

![image](https://github.com/user-attachments/assets/48402445-0843-45f5-b372-6a5865e68123)
![image](https://github.com/user-attachments/assets/df28d53f-6242-4b72-a771-b034edf36dc1)

进一步查询 CVE-2024-3568 :
![image](https://github.com/user-attachments/assets/10102e67-432c-4717-969e-93204960fc75)

Huggingface 的 transformers 库存在通过不可信数据反序列化导致的任意代码执行漏洞。该漏洞位于 TFPreTrainedModel() 类中的 load_repo_checkpoint() 函数，攻击者可以通过构造一个恶意的序列化负载，利用 pickle.load() 反序列化不受信任的数据，从而执行任意代码和命令。攻击者可以通过在正常的训练过程中欺骗受害者加载看似无害的检查点（checkpoint），从而在目标机器上执行任意代码，导致远程代码执行（RCE）漏洞。

*Checkpoint（检查点）是机器学习和深度学习中的一个概念，通常指的是模型在训练过程中某个时间点的“快照”。这个快照保存了模型的权重（weights）、优化器状态（optimizer state）以及其他可能的重要信息（如当前的训练进度、学习率等）。通过保存这些信息，我们可以在后续需要的时候恢复训练，或者用来评估模型的表现，而无需从头重新训练。

pickle 反序列化的漏洞利用原理是基于 pickle 模块的工作方式。在 Python 中，pickle 是一个用于序列化和反序列化对象的模块，能够将对象序列化成字节流并存储或传输，然后通过反序列化将字节流还原为对象。问题在于 pickle 的反序列化过程中不仅仅是简单的还原对象，它还会执行其中的代码。如果 pickle 数据流中包含了恶意代码，那么反序列化时会执行这些代码，导致任意代码执行漏洞。

工作原理：
序列化：在使用 pickle 模块序列化对象时，Python 会将对象的属性、状态等打包成字节流。在此过程中，pickle 还会保存一些关于如何构造对象的元信息，包括可能调用的构造函数或方法。

反序列化：当我们使用 pickle.load() 或 pickle.loads() 来反序列化对象时，pickle 会根据这些元信息重建对象。这意味着，它会调用对象的构造函数和某些方法来重建对象的状态。

漏洞点：正是因为 pickle 在反序列化时会执行与对象构建相关的代码，攻击者可以将恶意代码嵌入序列化的字节流中。一旦不小心从不受信任的来源反序列化数据，攻击者的恶意代码就会在目标机器上被执行。

去看huggingface transformer，可以直接看哪个transformer 版本有load_repo_checkpoint()。发现V4.17.0 用到load_repo_checkpoint。

V4.17.0版本

![image](https://github.com/user-attachments/assets/708deab1-82a3-4e92-8c48-39af89926116)

最新版本已经被修复 

尝试写POC (此处用AI 快速生成脚本) :
![image](https://github.com/user-attachments/assets/cb4000f3-fa92-424c-af1e-b746bc0c5384)
![image](https://github.com/user-attachments/assets/a6620e52-1d39-497d-88e8-ffe92098d23e)
![image](https://github.com/user-attachments/assets/43ea26ae-93bb-4924-a2aa-929e3229cf48)



此处为恶意代码的注入，暂时用hello world 代替 :

![image](https://github.com/user-attachments/assets/58569b52-7435-4a09-8425-d390f89a1bc9)


更多大瓜 :

![image](https://github.com/user-attachments/assets/a29c1fd2-ce47-4bda-a255-ca604c6ec33a)

