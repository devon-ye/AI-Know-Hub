import type {SidebarConfig} from '@vuepress/theme-default'

export const sidebarZh: SidebarConfig = {
    // '/theoretical-foundations/': [
    //   {
    //     text: '理论基础',
    //     children: [
    //       '/theoretical-foundations/mathematical-foundations.md',
    //       '/theoretical-foundations/computational-models.md',
    //       '/theoretical-foundations/cognitive-science-basics.md',
    //     ],
    //   },
    // ],
    // '/core-technologies': [
    //   {
    //     text: '机器学习',
    //     children: [
    //       '/core-technologies/machine-learning/introduction.md',
    //       '/core-technologies/machine-learning/supervised-learning.md',
    //       '/core-technologies/machine-learning/unsupervised-learning.md',
    //       '/core-technologies/machine-learning/reinforcement-learning.md',
    //     ],
    //   },
    //   {
    //     text: '深度学习',
    //     children: [
    //
    //     ],
    //   },
    // ],
    // '/application-domains/': [
    //   {
    //     text: '自然语言处理',
    //     collapsible: true,
    //     children: [
    //
    //     ],
    //   },
    //   {
    //     text: '计算视觉',
    //     collapsible: true,
    //     children: [
    //
    //     ],
    //   },
    //   {
    //     text: '机器人学',
    //     collapsible: true,
    //     children: [
    //
    //     ],
    //   },],
    '/engineering-practices/': [
        {
            text: '工程化实践',
            collapsible: true,
            children: [
                {
                    text: '指南',
                    children: [
                        '/engineering-practices/introduction.md',
                        '/engineering-practices/get-started.md',
                    ],
                },
                {
                    text: '接口',
                    children: [
                    ],
                },
                {
                    text: '方案',
                    children: [],
                },
                {
                    text: '模型部署',
                    children: [],
                },
                {
                    text: '模型微调',
                    children: [
                        '/engineering-practices/model-fine-tuning/introduction.md',
                        '/engineering-practices/model-fine-tuning/Prompt-Tuning.md',
                        '/engineering-practices/model-fine-tuning/LoRA-Tuning.md',
                    ],
                },
                {
                    text: '模型评估',
                    children: [],
                }
            ],
        },
    ],
}
