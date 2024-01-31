import type { NavbarConfig } from '@vuepress/theme-default'
import { version } from '../meta.js'

export const navbarEn: NavbarConfig = [
  {
    text: 'Theoretical Foundations',
    children: [
      '/en/theoretical-foundations/mathematical-foundations.md',
      '/en/theoretical-foundations/computational-models.md',
      '/en/theoretical-foundations/cognitive-science-basics.md',
    ],
  },
  {
    text: 'Core Technologies',
    children: [
      {
        text: 'Machine Learning',
        children: [
          '/core-technologies/machine-learning/introduction.md',
          '/core-technologies/machine-learning/supervised-learning.md',
          '/core-technologies/machine-learning/unsupervised-learning.md',
          '/core-technologies/machine-learning/reinforcement-learning.md',
        ],
      },
      {
        text: 'Deep Learning',
        children: [
        ],
      }
    ],
  },
  {
    text: 'Application Domains',
    children: [
      {
        text: 'NLP',
        children: [
        ],
      },
      {
        text: 'CV',
        children: [

        ],
      },
      {
        text: 'Robotics',
        children: [

        ],
      }
    ],
  },
  {
    text: 'Engineering Practice',
    children: [
      {
        text: 'Model Training',
        children: [
        ],
      },
      {
        text: 'Model Evaluation',
        children: [
        ],
      },
    ],
  },
  {
    text: `Large Model Security`,
    children: [
    ],
  },
  {
    text: `Frontiers and Ethics`,
    children: [

    ],
  },
]
