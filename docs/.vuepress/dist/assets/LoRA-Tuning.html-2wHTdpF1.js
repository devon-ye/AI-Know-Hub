import{_ as s,o as n,c as a,a as l}from"./app-ECqxxh7V.js";const e={},i=l(`<h1 id="lora" tabindex="-1"><a class="header-anchor" href="#lora"><span>LoRA</span></a></h1><p>Lora微调实际上是指一种特定的模型微调技术，称为&quot;LoRA&quot;，全称为&quot;Low-Rank Adaptation&quot;（低秩适配）。核心思想是在模型的预训练权重基础上，</p><p>通过引入额外的、较小的、可训练的参数矩阵来实现微调，这些矩阵作为原始权重的低秩更新。</p><h2 id="原理" tabindex="-1"><a class="header-anchor" href="#原理"><span>原理</span></a></h2><h2 id="流程" tabindex="-1"><a class="header-anchor" href="#流程"><span>流程</span></a></h2><ul><li><p>准备数据集</p></li><li><p>选择预训练模型</p></li><li><p>构建模型</p></li><li><p>定义损失函数</p></li><li><p>定义优化器</p></li><li><p>训练模型</p></li><li><p>评估模型</p></li></ul><h2 id="实战" tabindex="-1"><a class="header-anchor" href="#实战"><span>实战</span></a></h2><h3 id="依赖库导入" tabindex="-1"><a class="header-anchor" href="#依赖库导入"><span>依赖库导入</span></a></h3><div class="language-bash line-numbers-mode" data-ext="sh" data-title="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"><span style="color:#DCDCAA;">pip</span><span style="color:#CE9178;"> install -q peft transformers datasets</span></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="准备数据集" tabindex="-1"><a class="header-anchor" href="#准备数据集"><span>准备数据集</span></a></h3><ul><li>加载数据集</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> datasets </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> load_dataset   </span><span style="color:#6A9955;"># 导入数据集加载函数</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">ds= load_dataset(</span><span style="color:#CE9178;">&#39;imdb&#39;</span><span style="color:#D4D4D4;">)           </span><span style="color:#6A9955;"># 加载IMDB数据集</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>数据集预处理</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">labels = ds[</span><span style="color:#CE9178;">&quot;train&quot;</span><span style="color:#D4D4D4;">].features[</span><span style="color:#CE9178;">&quot;label&quot;</span><span style="color:#D4D4D4;">].names</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">label2id, id2label = </span><span style="color:#4EC9B0;">dict</span><span style="color:#D4D4D4;">(), </span><span style="color:#4EC9B0;">dict</span><span style="color:#D4D4D4;">()</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">for</span><span style="color:#D4D4D4;"> i, label </span><span style="color:#C586C0;">in</span><span style="color:#DCDCAA;"> enumerate</span><span style="color:#D4D4D4;">(labels):</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    label2id[label] = i</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    id2label[i] = label</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">id2label[</span><span style="color:#B5CEA8;">2</span><span style="color:#D4D4D4;">]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#CE9178;">&quot;baklava&quot;</span></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>特征缩放标准化</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> transformers </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> AutoImageProcessor</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">image_processor = AutoImageProcessor.from_pretrained(</span><span style="color:#CE9178;">&quot;google/vit-base-patch16-224-in21k&quot;</span><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> torchvision.transforms </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> (</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    CenterCrop,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    Compose,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    Normalize,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    RandomHorizontalFlip,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    RandomResizedCrop,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    Resize,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    ToTensor,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">normalize = Normalize(</span><span style="color:#9CDCFE;">mean</span><span style="color:#D4D4D4;">=image_processor.image_mean, </span><span style="color:#9CDCFE;">std</span><span style="color:#D4D4D4;">=image_processor.image_std)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">train_transforms = Compose(</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    [</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        RandomResizedCrop(image_processor.size[</span><span style="color:#CE9178;">&quot;height&quot;</span><span style="color:#D4D4D4;">]),</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        RandomHorizontalFlip(),</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        ToTensor(),</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        normalize,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    ]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">val_transforms = Compose(</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    [</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        Resize(image_processor.size[</span><span style="color:#CE9178;">&quot;height&quot;</span><span style="color:#D4D4D4;">]),</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        CenterCrop(image_processor.size[</span><span style="color:#CE9178;">&quot;height&quot;</span><span style="color:#D4D4D4;">]),</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        ToTensor(),</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">        normalize,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    ]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#DCDCAA;"> preprocess_train</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">example_batch</span><span style="color:#D4D4D4;">):</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    example_batch[</span><span style="color:#CE9178;">&quot;pixel_values&quot;</span><span style="color:#D4D4D4;">] = [train_transforms(image.convert(</span><span style="color:#CE9178;">&quot;RGB&quot;</span><span style="color:#D4D4D4;">)) </span><span style="color:#C586C0;">for</span><span style="color:#D4D4D4;"> image </span><span style="color:#C586C0;">in</span><span style="color:#D4D4D4;"> example_batch[</span><span style="color:#CE9178;">&quot;image&quot;</span><span style="color:#D4D4D4;">]]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">    return</span><span style="color:#D4D4D4;"> example_batch</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#DCDCAA;"> preprocess_val</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">example_batch</span><span style="color:#D4D4D4;">):</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    example_batch[</span><span style="color:#CE9178;">&quot;pixel_values&quot;</span><span style="color:#D4D4D4;">] = [val_transforms(image.convert(</span><span style="color:#CE9178;">&quot;RGB&quot;</span><span style="color:#D4D4D4;">)) </span><span style="color:#C586C0;">for</span><span style="color:#D4D4D4;"> image </span><span style="color:#C586C0;">in</span><span style="color:#D4D4D4;"> example_batch[</span><span style="color:#CE9178;">&quot;image&quot;</span><span style="color:#D4D4D4;">]]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">    return</span><span style="color:#D4D4D4;"> example_batch</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>数据集划分</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">train_ds = ds[</span><span style="color:#CE9178;">&quot;train&quot;</span><span style="color:#D4D4D4;">]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">val_ds = ds[</span><span style="color:#CE9178;">&quot;validation&quot;</span><span style="color:#D4D4D4;">]</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">train_ds.set_transform(preprocess_train)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">val_ds.set_transform(preprocess_val)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>微调数据整理器</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> torch</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#DCDCAA;"> collate_fn</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">examples</span><span style="color:#D4D4D4;">):</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    pixel_values = torch.stack([example[</span><span style="color:#CE9178;">&quot;pixel_values&quot;</span><span style="color:#D4D4D4;">] </span><span style="color:#C586C0;">for</span><span style="color:#D4D4D4;"> example </span><span style="color:#C586C0;">in</span><span style="color:#D4D4D4;"> examples])</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    labels = torch.tensor([example[</span><span style="color:#CE9178;">&quot;label&quot;</span><span style="color:#D4D4D4;">] </span><span style="color:#C586C0;">for</span><span style="color:#D4D4D4;"> example </span><span style="color:#C586C0;">in</span><span style="color:#D4D4D4;"> examples])</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">    return</span><span style="color:#D4D4D4;"> {</span><span style="color:#CE9178;">&quot;pixel_values&quot;</span><span style="color:#D4D4D4;">: pixel_values, </span><span style="color:#CE9178;">&quot;labels&quot;</span><span style="color:#D4D4D4;">: labels}</span></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="选择预训练模型" tabindex="-1"><a class="header-anchor" href="#选择预训练模型"><span>选择预训练模型</span></a></h3><ul><li>构建模型</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> transformers </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> AutoModelForImageClassification, TrainingArguments, Trainer</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">model = AutoModelForImageClassification.from_pretrained(</span></span>
<span class="line"></span>
<span class="line"><span style="color:#CE9178;">    &quot;google/vit-base-patch16-224-in21k&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    label2id</span><span style="color:#D4D4D4;">=label2id,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    id2label</span><span style="color:#D4D4D4;">=id2label,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    ignore_mismatched_sizes</span><span style="color:#D4D4D4;">=</span><span style="color:#569CD6;">True</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>PEFT微调器配置</li></ul><h2 id="训练" tabindex="-1"><a class="header-anchor" href="#训练"><span>训练</span></a></h2><ul><li>定义损失函数</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> transformers </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> TrainingArguments, Trainer</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">account = </span><span style="color:#CE9178;">&quot;stevhliu&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">peft_model_id = </span><span style="color:#569CD6;">f</span><span style="color:#CE9178;">&quot;</span><span style="color:#569CD6;">{</span><span style="color:#D4D4D4;">account</span><span style="color:#569CD6;">}</span><span style="color:#CE9178;">/google/vit-base-patch16-224-in21k-lora&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">batch_size = </span><span style="color:#B5CEA8;">128</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">args = TrainingArguments(</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    peft_model_id,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    remove_unused_columns</span><span style="color:#D4D4D4;">=</span><span style="color:#569CD6;">False</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    evaluation_strategy</span><span style="color:#D4D4D4;">=</span><span style="color:#CE9178;">&quot;epoch&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    save_strategy</span><span style="color:#D4D4D4;">=</span><span style="color:#CE9178;">&quot;epoch&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    learning_rate</span><span style="color:#D4D4D4;">=</span><span style="color:#B5CEA8;">5e-3</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    per_device_train_batch_size</span><span style="color:#D4D4D4;">=batch_size,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    gradient_accumulation_steps</span><span style="color:#D4D4D4;">=</span><span style="color:#B5CEA8;">4</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    per_device_eval_batch_size</span><span style="color:#D4D4D4;">=batch_size,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    fp16</span><span style="color:#D4D4D4;">=</span><span style="color:#569CD6;">True</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    num_train_epochs</span><span style="color:#D4D4D4;">=</span><span style="color:#B5CEA8;">5</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    logging_steps</span><span style="color:#D4D4D4;">=</span><span style="color:#B5CEA8;">10</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    load_best_model_at_end</span><span style="color:#D4D4D4;">=</span><span style="color:#569CD6;">True</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    label_names</span><span style="color:#D4D4D4;">=[</span><span style="color:#CE9178;">&quot;labels&quot;</span><span style="color:#D4D4D4;">],</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>定义优化器</li></ul><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">trainer = Trainer(</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    model,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    args,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    train_dataset</span><span style="color:#D4D4D4;">=train_ds,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    eval_dataset</span><span style="color:#D4D4D4;">=val_ds,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    tokenizer</span><span style="color:#D4D4D4;">=image_processor,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    data_collator</span><span style="color:#D4D4D4;">=collate_fn,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">trainer.train()</span></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="评估" tabindex="-1"><a class="header-anchor" href="#评估"><span>评估</span></a></h2><h2 id="模型发布" tabindex="-1"><a class="header-anchor" href="#模型发布"><span>模型发布</span></a></h2><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> huggingface_hub </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> notebook_login</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">notebook_login()</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">model.push_to_hub(peft_model_id)</span></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="模型部署" tabindex="-1"><a class="header-anchor" href="#模型部署"><span>模型部署</span></a></h2><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;color:#D4D4D4;" tabindex="0"><code><span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> peft </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> PeftConfig, PeftModel</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> transfomers </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> AutoImageProcessor</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> PIL </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> Image</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> requests</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">config = PeftConfig.from_pretrained(</span><span style="color:#CE9178;">&quot;stevhliu/vit-base-patch16-224-in21k-lora&quot;</span><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">model = AutoModelForImageClassification.from_pretrained(</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">    config.base_model_name_or_path,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    label2id</span><span style="color:#D4D4D4;">=label2id,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    id2label</span><span style="color:#D4D4D4;">=id2label,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#9CDCFE;">    ignore_mismatched_sizes</span><span style="color:#D4D4D4;">=</span><span style="color:#569CD6;">True</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">model = PeftModel.from_pretrained(model, </span><span style="color:#CE9178;">&quot;stevhliu/vit-base-patch16-224-in21k-lora&quot;</span><span style="color:#D4D4D4;">)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">url = </span><span style="color:#CE9178;">&quot;https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">image = Image.open(requests.get(url, </span><span style="color:#9CDCFE;">stream</span><span style="color:#D4D4D4;">=</span><span style="color:#569CD6;">True</span><span style="color:#D4D4D4;">).raw)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#D4D4D4;">image</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,34),p=[i];function c(o,r){return n(),a("div",null,p)}const t=s(e,[["render",c],["__file","LoRA-Tuning.html.vue"]]);export{t as default};
