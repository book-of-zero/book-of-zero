// Code copy button and language label functionality
(function() {
  'use strict';

  function addCopyButtons() {
    // Find all code blocks
    const codeBlocks = document.querySelectorAll('div.highlight, pre.highlight');

    codeBlocks.forEach(function(codeBlock) {
      // Skip if already processed
      if (codeBlock.parentNode.classList.contains('code-block-wrapper')) {
        return;
      }

      // Create wrapper
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      // Wrap the code block
      codeBlock.parentNode.insertBefore(wrapper, codeBlock);
      wrapper.appendChild(codeBlock);

      // Extract language from class (e.g., "language-python" or "highlight-python")
      let language = '';
      const classes = codeBlock.className.split(' ');
      for (let cls of classes) {
        if (cls.startsWith('language-')) {
          language = cls.replace('language-', '').toUpperCase();
          break;
        } else if (cls.startsWith('highlight-')) {
          language = cls.replace('highlight-', '').toUpperCase();
          break;
        }
      }

      // Add language label if found
      if (language) {
        const languageLabel = document.createElement('div');
        languageLabel.className = 'code-language-label';
        languageLabel.textContent = language;
        wrapper.insertBefore(languageLabel, codeBlock);
      }

      // Create copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-code-button';
      copyButton.textContent = 'Copy';
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');

      // Add copy functionality
      copyButton.addEventListener('click', function() {
        // Find the code element
        const code = codeBlock.querySelector('code') || codeBlock.querySelector('pre');
        const textToCopy = code ? code.textContent : codeBlock.textContent;

        // Copy to clipboard
        navigator.clipboard.writeText(textToCopy).then(function() {
          // Success feedback
          copyButton.textContent = 'Copied!';
          copyButton.classList.add('copied');

          // Reset after 2 seconds
          setTimeout(function() {
            copyButton.textContent = 'Copy';
            copyButton.classList.remove('copied');
          }, 2000);
        }).catch(function(err) {
          console.error('Failed to copy code: ', err);
          copyButton.textContent = 'Error';
          setTimeout(function() {
            copyButton.textContent = 'Copy';
          }, 2000);
        });
      });

      wrapper.appendChild(copyButton);
    });
  }

  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addCopyButtons);
  } else {
    addCopyButtons();
  }
})();
