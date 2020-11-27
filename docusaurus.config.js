const math = require('remark-math')
const katex = require('rehype-katex')

module.exports = {
  title: 'carefree-learn',
  url: 'https://carefree0910.me',
  baseUrl: '/carefree-learn-doc/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'carefree0910', // Usually your GitHub org/user name.
  projectName: 'carefree-learn-doc', // Usually your repo name.
  stylesheets: [
    'https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css',
  ],
  themeConfig: {
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
    },
    announcementBar: {
      id: 'supportus',
      content:
        '⭐️ If you like carefree-learn, give it a star on <a target="_blank" href="https://github.com/carefree0910/carefree-learn">GitHub</a>! ⭐️',
    },
    algolia: {
      apiKey: '42cba8d66353791f378a6fbc4d8d05cf',
      indexName: 'carefree-learn',
      contextualSearch: true,
    },
    navbar: {
      title: 'carefree-learn',
      items: [
        {
          to: 'docs/',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        {to: 'blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/carefree0910/carefree-learn',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Installation',
              to: 'docs/getting-started/installation/',
            },
            {
              label: 'Quick Start',
              to: 'docs/getting-started/quick-start',
            },
            {
              label: 'Design Principles',
              to: 'docs/design-principles/',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Twitter',
              href: 'https://twitter.com/carefree0910',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: 'blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/carefree0910/carefree-learn',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} carefree-learn, carefree0910. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/edit/master/website/',
          showLastUpdateTime: true,
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/edit/master/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
