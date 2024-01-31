import type { SidebarConfig } from '@vuepress/theme-default'

export const sidebarZh: SidebarConfig = {
  '/theoretical-foundations/': [
    {
      text: '理论基础',
      children: [
        '/theoretical-foundations/mathematical-foundations.md',
        '/theoretical-foundations/computational-models.md',
        '/theoretical-foundations/cognitive-science-basics.md',
      ],
    },
  ],
  '/core-technologies': [
    {
      text: '机器学习',
      children: [
        '/core-technologies/machine-learning/introduction.md',
        '/core-technologies/machine-learning/supervised-learning.md',
        '/core-technologies/machine-learning/unsupervised-learning.md',
        '/core-technologies/machine-learning/reinforcement-learning.md',
      ],
    },
    {
      text: '深度学习',
      children: [

      ],
    },
  ],
  '/application-domains/': [
    {
      text: '自然语言处理',
      collapsible: true,
      children: [

      ],
    },
    {
      text: '计算视觉',
      collapsible: true,
      children: [

      ],
    },
    {
      text: '机器人学',
      collapsible: true,
      children: [

      ],
    },
    {
      text: '工程化实践',
      collapsible: true,
      children: [
        {
          text: '模型训练',
          children: [
          ],
        },
        {
          text: '模型评估',
          children: [

          ],
        }
      ],
    },
  ],
}
