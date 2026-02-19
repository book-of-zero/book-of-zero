require(["gitbook", "jquery"], function (gitbook, $) {
  // Current fontsettings state
  var fontState;
  var BUTTON_ID;

  // Only allow Light + Dark
  // Keep ids aligned with GitBook conventions:
  // - 0: default (light)
  // - 2: night (dark)
  var THEMES = [
    { config: "white", text: "Light", id: 0 },
    { config: "night", text: "Dark", id: 2 },
  ];

  // Force Sans only (prevents serif from being set via storage)
  var FAMILY_SANS_ID = 1;
  var SIZE_DEFAULT_ID = 2;

  function getThemeId(configName) {
    var configTheme = $.grep(THEMES, function (theme) {
      return theme.config === configName;
    })[0];
    return configTheme ? configTheme.id : 0;
  }

  function saveFontSettings() {
    gitbook.storage.set("fontState", fontState);
    update();
  }

  function changeColorTheme(configName, e) {
    if (e && typeof e.preventDefault === "function") e.preventDefault();

    var $book = gitbook.state.$book;
    var $header = $(".book-body > .book-header");

    // Remove any previously applied color theme
    $book[0].className = $book[0].className.replace(/\bcolor-theme-\S+/g, "");
    if ($header.length !== 0) {
      $header[0].className = $header[0].className.replace(/\bcolor-theme-\S+/g, "");
    }

    // Apply new color theme
    var themeId = getThemeId(configName);
    fontState.theme = themeId === 1 ? 0 : themeId; // hard-block sepia

    if (fontState.theme !== 0) {
      $book.addClass("color-theme-" + fontState.theme);
      if ($header.length !== 0) $header.addClass("color-theme-" + fontState.theme);
    }

    saveFontSettings();
  }

  function update() {
    var $book = gitbook.state.$book;
    var $header = $(".book-body > .book-header");

    // Lock font size + family (no serif / no font-size UI)
    fontState.size = SIZE_DEFAULT_ID;
    fontState.family = FAMILY_SANS_ID;

    $book[0].className = $book[0].className
      .replace(/\bfont-size-\S+/g, "")
      .replace(/\bfont-family-\S+/g, "")
      .replace(/\bcolor-theme-\S+/g, "");

    $book.addClass("font-size-" + fontState.size);
    $book.addClass("font-family-" + fontState.family);

    if (fontState.theme !== 0) $book.addClass("color-theme-" + fontState.theme);

    if ($header.length !== 0) {
      $header[0].className = $header[0].className.replace(/\bcolor-theme-\S+/g, "");
      if (fontState.theme !== 0) $header.addClass("color-theme-" + fontState.theme);
    }
  }

  function init(config) {
    var opts = config || {};
    var configTheme = getThemeId(opts.theme);

    var stored = gitbook.storage.get("fontState", {});
    var storedTheme = stored && typeof stored.theme === "number" ? stored.theme : null;
    var theme = storedTheme === 1 ? 0 : storedTheme; // normalize sepia -> light
    if (typeof theme !== "number") theme = configTheme;

    fontState = {
      size: SIZE_DEFAULT_ID,
      family: FAMILY_SANS_ID,
      theme: theme || 0,
    };

    update();
    gitbook.storage.set("fontState", fontState);
  }

  function updateButtons() {
    // Remove existing fontsettings button
    if (BUTTON_ID) gitbook.toolbar.removeButton(BUTTON_ID);

    BUTTON_ID = gitbook.toolbar.createButton({
      icon: "fa fa-adjust",
      label: "Theme",
      className: "font-settings",
      dropdown: [
        [
          $.map(THEMES, function (theme) {
            return {
              text: theme.text,
              className: theme.config,
              onClick: function (e) {
                return changeColorTheme(theme.config, e);
              },
            };
          }),
        ],
      ],
    });
  }

  gitbook.events.bind("start", function (e, config) {
    updateButtons();
    init((config && config.fontsettings) || {});
  });

  // Expose minimal API
  gitbook.fontsettings = gitbook.fontsettings || {};
  gitbook.fontsettings.setTheme = changeColorTheme;
});

