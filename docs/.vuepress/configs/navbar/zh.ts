import type {NavbarConfig} from '@vuepress/theme-default'
import {version} from '../meta.js'

export const navbarZh: NavbarConfig = [
    // {
    //   text: '理论基础',
    //   children: [
    //     '/theoretical-foundations/mathematical-foundations.md',
    //     '/theoretical-foundations/computational-models.md',
    //     '/theoretical-foundations/cognitive-science-basics.md',
    //   ],
    // },
    // {
    //   text: '核心技术',
    //   children: [
    //     {
    //       text: '机器学习',
    //       children: [
    //          '/core-technologies/machine-learning/introduction.md',
    //          '/core-technologies/machine-learning/supervised-learning.md',
    //          '/core-technologies/machine-learning/unsupervised-learning.md',
    //          '/core-technologies/machine-learning/reinforcement-learning.md',
    //       ],
    //     },
    //     {
    //       text: '深度学习',
    //       children: [
    //       ],
    //     }
    //   ],
    // },
    // {
    //   text: '应用领域',
    //   children: [
    //     {
    //       text: '自然语言处理',
    //       children: [
    //       ],
    //     },
    //     {
    //       text: '计算视觉',
    //       children: [
    //
    //       ],
    //     },
    //     {
    //       text: '机器人学',
    //       children: [
    //
    //       ],
    //     }
    //   ],
    // },
    {
        text: '工程化实践',
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
    // {
    //   text: `大模型安全`,
    //   children: [
    //   ],
    // },
    // {
    //   text: `前沿与伦理`,
    //   children: [
    //
    //   ],
    // },
]
