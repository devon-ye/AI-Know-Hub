import type {SidebarConfig} from '@vuepress/theme-default'

export const sidebarEn: SidebarConfig = {
    // '/theoretical-foundations/': [
    //   {
    //     text: 'Theoretical Foundations',
    //     children: [
    //       '/en/theoretical-foundations/mathematical-foundations.md',
    //       '/en/theoretical-foundations/computational-models.md',
    //       '/en/theoretical-foundations/cognitive-science-basics.md',
    //     ],
    //   },
    // ],
    // '/core-technologies': [
    //   {
    //     text: 'Core Technologies',
    //     children: [
    //       '/en/core-technologies/machine-learning/introduction.md',
    //       '/en/core-technologies/machine-learning/supervised-learning.md',
    //       '/en/core-technologies/machine-learning/unsupervised-learning.md',
    //       '/en/core-technologies/machine-learning/reinforcement-learning.md',
    //     ],
    //   },
    //   {
    //     text: 'Deep learning',
    //     children: [
    //
    //     ],
    //   },
    // ],
    // '/application-domains/': [
    //     {
    //         text: 'NLP',
    //         collapsible: true,
    //         children: [],
    //     },
    //     {
    //         text: 'CV',
    //         collapsible: true,
    //         children: [],
    //     },
    //     {
    //         text: 'Robotics',
    //         collapsible: true,
    //         children: [],
    //     },
    // ],
    "Engineering Practice": [{
        text: 'Engineering Practice',
        collapsible: true,
        children: [
            {
                text: '开始',
                children: [
                    '/engineering-practices/introduction.md',
                ],
            },
            {
                text: 'Model Training',
                children: [],
            },
            {
                text: 'Model Fine-Tuning',
                children: [],
            },

            {
                text: 'Model Evaluation',
                children: [],
            }
        ],
    },
    ],
}
