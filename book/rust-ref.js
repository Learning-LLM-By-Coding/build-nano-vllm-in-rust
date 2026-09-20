// Open reference links in a side panel instead of navigating away, so the
// background reading sits right next to the code that uses it. Sources:
// Google's Comprehensive Rust (Rust concepts; offline-stageable via
// scripts/vendor-rust-ref.sh) plus Dive into Deep Learning and labml.ai's
// annotated implementations (deep-learning concepts; online only).
// Plain left-clicks are intercepted; middle/modified clicks and the panel's
// own "open in new tab" link still behave like normal links.
(function () {
    "use strict";

    var SOURCES = [
        {
            prefix: "https://google.github.io/comprehensive-rust/",
            label: "Comprehensive Rust",
            localBase:
                (typeof path_to_root !== "undefined" ? path_to_root : "") +
                "comprehensive-rust/",
        },
        { prefix: "https://d2l.ai/", label: "Dive into Deep Learning" },
        { prefix: "https://nn.labml.ai/", label: "labml.ai annotated papers" },
        { prefix: "https://www.youtube.com/watch", label: "Video", video: true },
        { prefix: "https://youtu.be/", label: "Video", video: true },
    ];
    var useLocal = false;
    var panel = null;
    var frame = null;
    var openLink = null;
    var titleEl = null;

    // Probe once for the vendored Comprehensive Rust copy; the vendor script
    // drops a vendored.ok marker next to the pages.
    try {
        fetch(SOURCES[0].localBase + "vendored.ok", { method: "HEAD" })
            .then(function (r) {
                useLocal = r.ok;
            })
            .catch(function () {
                useLocal = false;
            });
    } catch (e) {
        // file:// or a very old browser: keep using the live sites.
    }

    function sourceFor(href) {
        for (var i = 0; i < SOURCES.length; i++) {
            if (href.indexOf(SOURCES[i].prefix) === 0) {
                return SOURCES[i];
            }
        }
        return null;
    }

    function liveUrl(source, url) {
        if (source.video) {
            // YouTube watch pages refuse to be framed; the embed player
            // doesn't. The new-tab link keeps the canonical watch URL.
            var m =
                url.match(/[?&]v=([A-Za-z0-9_-]{6,})/) ||
                url.match(/youtu\.be\/([A-Za-z0-9_-]{6,})/);
            if (m) {
                return "https://www.youtube-nocookie.com/embed/" + m[1];
            }
        }
        return url;
    }

    function localCandidate(source, url) {
        if (!source.localBase || !useLocal || source.video) {
            return null;
        }
        var rest = url.slice(source.prefix.length);
        return source.localBase + (rest === "" ? "index.html" : rest);
    }

    function ensurePanel() {
        if (panel) {
            return;
        }
        panel = document.createElement("aside");
        panel.id = "rust-ref-panel";

        var bar = document.createElement("div");
        bar.className = "rust-ref-bar";

        titleEl = document.createElement("span");
        titleEl.className = "rust-ref-title";

        openLink = document.createElement("a");
        openLink.textContent = "open in new tab ↗";
        openLink.target = "_blank";
        openLink.rel = "noopener";

        var close = document.createElement("button");
        close.className = "rust-ref-close";
        close.setAttribute("aria-label", "Close reference panel");
        close.textContent = "×";
        close.addEventListener("click", closePanel);

        bar.appendChild(titleEl);
        bar.appendChild(openLink);
        bar.appendChild(close);

        frame = document.createElement("iframe");
        frame.title = "Reference";

        panel.appendChild(bar);
        panel.appendChild(frame);
        document.body.appendChild(panel);
    }

    var openSeq = 0;

    function openPanel(source, url) {
        ensurePanel();
        titleEl.textContent = source.label;
        // Sharing and "read more" stay on the canonical live site.
        openLink.href = url;
        document.body.classList.add("rust-ref-open");
        var live = liveUrl(source, url);
        var local = localCandidate(source, url);
        if (!local) {
            frame.src = live;
            return;
        }
        // A staged copy is a snapshot of whatever the chapters linked when
        // it was made, so a later chapter can link a page it lacks. Probe
        // for this exact page and fall back to the live site rather than
        // showing the book's own 404 inside the panel.
        var seq = ++openSeq;
        fetch(local, { method: "HEAD" })
            .then(function (r) {
                if (seq === openSeq) {
                    frame.src = r.ok ? local : live;
                }
            })
            .catch(function () {
                if (seq === openSeq) {
                    frame.src = live;
                }
            });
    }

    function closePanel() {
        document.body.classList.remove("rust-ref-open");
        if (frame) {
            frame.src = "about:blank";
        }
    }

    document.addEventListener("click", function (e) {
        if (e.defaultPrevented || e.button !== 0) {
            return;
        }
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
            return;
        }
        var a = e.target && e.target.closest ? e.target.closest("a[href]") : null;
        if (!a) {
            return;
        }
        var source = sourceFor(a.href);
        if (!source || a === openLink) {
            return;
        }
        e.preventDefault();
        openPanel(source, a.href);
    });

    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
            closePanel();
        }
    });
})();
