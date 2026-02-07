require(["gitbook", "jquery"], function (gitbook, $) {
  "use strict";

  function copyToClipboard(text) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      return navigator.clipboard.writeText(text);
    }

    if (document.queryCommandSupported && document.queryCommandSupported("copy")) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.top = "-9999px";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand("copy");
        return Promise.resolve();
      } catch (e) {
        return Promise.reject(e);
      } finally {
        document.body.removeChild(textarea);
      }
    }

    return Promise.reject(new Error("Clipboard API unavailable"));
  }

  function iconCopy() {
    // Heroicons-style "copy" icon (inline SVG)
    return (
      '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
      '<path d="M9 9h10v10H9V9Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>' +
      '<path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>' +
      "</svg>"
    );
  }

  function iconCheck() {
    return (
      '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
      '<path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>' +
      "</svg>"
    );
  }

  function iconX() {
    return (
      '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
      '<path d="M18 6 6 18M6 6l12 12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>' +
      "</svg>"
    );
  }

  function setButtonState($btn, state) {
    $btn.attr("data-state", state || "");

    if (state === "copied") {
      $btn.attr("title", "Copied");
      $btn.attr("aria-label", "Copied");
      $btn.html(iconCheck());
      return;
    }

    if (state === "error") {
      $btn.attr("title", "Copy failed");
      $btn.attr("aria-label", "Copy failed");
      $btn.html(iconX());
      return;
    }

    $btn.attr("title", "Copy");
    $btn.attr("aria-label", "Copy code");
    $btn.html(iconCopy());
  }

  function getCodeText($pre) {
    var $code = $pre.children("code");
    if ($code.length) return $code.text();
    return $pre.text();
  }

  gitbook.events.bind("page.change", function () {
    $("pre").each(function () {
      var $pre = $(this);

      // Only add for code blocks
      if ($pre.children("code").length === 0) return;

      // Avoid duplicating on SPA navigation
      if ($pre.children(".boz-copy-code").length) return;

      var $btn = $('<button type="button" class="boz-copy-code"></button>');
      setButtonState($btn, "");

      $btn.on("click", function (e) {
        e.preventDefault();
        e.stopPropagation();

        var text = getCodeText($pre);
        setButtonState($btn, ""); // reset icon immediately

        copyToClipboard(text)
          .then(function () {
            setButtonState($btn, "copied");
            window.setTimeout(function () {
              setButtonState($btn, "");
            }, 1200);
          })
          .catch(function () {
            setButtonState($btn, "error");
            window.setTimeout(function () {
              setButtonState($btn, "");
            }, 1500);
          });
      });

      $pre.append($btn);
    });
  });
});

