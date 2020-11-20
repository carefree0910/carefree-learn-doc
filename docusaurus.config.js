const math = require('remark-math')
const katex = require('rehype-katex')

module.exports = {
  title: 'carefree-learn',
  url: 'https://carefree0910.me/carefree-learn-doc/',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'carefree0910', // Usually your GitHub org/user name.
  projectName: 'carefree-learn', // Usually your repo name.
  stylesheets: [
    'https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css',
  ],
  themeConfig: {
    prism: {
      theme: require('prism-react-renderer/themes/github'),
    },
    navbar: {
      title: 'carefree-learn',
      // logo: {
      //   alt: 'carefree-learn',
      //   src: 'img/logo.svg',
      // },
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
          label: 'GitHub',
          position: 'right',
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
