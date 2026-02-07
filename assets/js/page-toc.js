// Right-side table of contents with scroll spy
(function() {
  'use strict';

  function generateTOC() {
    // Only run on larger screens
    if (window.innerWidth <= 1400) {
      return;
    }

    // Find all H2 and H3 headings in main content
    const mainContent = document.querySelector('main') || document.querySelector('.main-content') || document.body;
    const headings = mainContent.querySelectorAll('h2[id], h3[id]');

    // Need at least 2 headings to show TOC
    if (headings.length < 2) {
      return;
    }

    // Create TOC container
    const tocContainer = document.createElement('aside');
    tocContainer.className = 'page-toc';
    tocContainer.setAttribute('aria-label', 'Table of contents');

    const tocTitle = document.createElement('div');
    tocTitle.className = 'page-toc-title';
    tocTitle.textContent = 'On this page';
    tocContainer.appendChild(tocTitle);

    const tocList = document.createElement('nav');
    tocList.className = 'page-toc-list';

    // Build TOC links
    headings.forEach(function(heading) {
      const link = document.createElement('a');
      link.href = '#' + heading.id;
      link.textContent = heading.textContent;
      link.className = 'page-toc-link';

      // Add appropriate class based on heading level
      if (heading.tagName === 'H2') {
        link.classList.add('page-toc-h2');
      } else if (heading.tagName === 'H3') {
        link.classList.add('page-toc-h3');
      }

      // Smooth scroll on click
      link.addEventListener('click', function(e) {
        e.preventDefault();
        heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, null, '#' + heading.id);
      });

      tocList.appendChild(link);
    });

    tocContainer.appendChild(tocList);
    document.body.appendChild(tocContainer);

    // Scroll spy functionality
    let currentActiveLink = null;

    function updateActiveLink() {
      let current = '';

      headings.forEach(function(heading) {
        const rect = heading.getBoundingClientRect();
        // Check if heading is in upper portion of viewport
        if (rect.top <= 100) {
          current = heading.id;
        }
      });

      if (current && current !== currentActiveLink) {
        // Remove previous active
        const prevActive = tocList.querySelector('.page-toc-link.active');
        if (prevActive) {
          prevActive.classList.remove('active');
        }

        // Add new active
        const newActive = tocList.querySelector('a[href="#' + current + '"]');
        if (newActive) {
          newActive.classList.add('active');
          currentActiveLink = current;
        }
      }
    }

    // Initial update
    updateActiveLink();

    // Update on scroll (throttled)
    let scrollTimeout;
    window.addEventListener('scroll', function() {
      if (scrollTimeout) {
        window.cancelAnimationFrame(scrollTimeout);
      }
      scrollTimeout = window.requestAnimationFrame(updateActiveLink);
    }, { passive: true });

    // Handle window resize
    let resizeTimeout;
    window.addEventListener('resize', function() {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(function() {
        if (window.innerWidth <= 1400 && tocContainer.parentNode) {
          tocContainer.parentNode.removeChild(tocContainer);
        }
      }, 250);
    });
  }

  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', generateTOC);
  } else {
    generateTOC();
  }
})();
