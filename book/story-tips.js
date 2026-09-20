// Concept reminders: a link from a chapter to story.html#<concept> opens that
// concept's entry from "The restaurant, at a glance" in a small floating card
// right where you are reading, instead of navigating away. Middle/modified
// clicks, links inside the card, and the story page itself behave like
// normal links; if the story page can't be fetched, the link navigates.
(function () {
    "use strict";

    var STORY = "story.html";
    var root = typeof path_to_root !== "undefined" ? path_to_root : "";
    var storyDoc = null;
    var card = null;
    var opener = null;

    function onStoryPage() {
        return location.pathname.slice(-STORY.length) === STORY;
    }

    function loadStory() {
        if (!storyDoc) {
            storyDoc = fetch(root + STORY)
                .then(function (r) {
                    if (!r.ok) throw new Error("story page unavailable");
                    return r.text();
                })
                .then(function (html) {
                    return new DOMParser().parseFromString(html, "text/html");
                });
            storyDoc.catch(function () {
                storyDoc = null;
            });
        }
        return storyDoc;
    }

    function close() {
        if (!card) return;
        card.remove();
        card = null;
        if (opener) opener.focus();
        opener = null;
    }

    function place(anchor) {
        var r = anchor.getBoundingClientRect();
        var width = document.documentElement.clientWidth;
        var left = Math.min(r.left, width - card.offsetWidth - 12);
        card.style.left = window.scrollX + Math.max(12, left) + "px";
        card.style.top = window.scrollY + r.bottom + 8 + "px";
    }

    function show(anchor, entry, href) {
        close();
        card = document.createElement("div");
        card.className = "story-tip";
        card.setAttribute("role", "dialog");

        var closeBtn = document.createElement("button");
        closeBtn.className = "story-tip-close";
        closeBtn.setAttribute("aria-label", "Close reminder");
        closeBtn.textContent = "×";
        closeBtn.addEventListener("click", close);

        var body = document.createElement("div");
        body.className = "story-tip-body";
        body.innerHTML = entry.innerHTML;
        // Links in the entry are relative to the story page, at the book root.
        body.querySelectorAll("a[href]").forEach(function (a) {
            var h = a.getAttribute("href");
            if (h.charAt(0) === "#") a.setAttribute("href", root + STORY + h);
            else if (!/^([a-z]+:|\/)/i.test(h)) a.setAttribute("href", root + h);
        });

        var more = document.createElement("a");
        more.className = "story-tip-more";
        more.href = href;
        more.textContent = "See the whole story page →";

        card.appendChild(closeBtn);
        card.appendChild(body);
        card.appendChild(more);
        document.body.appendChild(card);
        place(anchor);
        opener = anchor;
        closeBtn.focus();
    }

    document.addEventListener("click", function (e) {
        var a = e.target.closest ? e.target.closest("a[href]") : null;
        if (card && card.contains(e.target)) return;
        if (!a) {
            close();
            return;
        }
        if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        if (onStoryPage() || !a.closest("main")) return;
        var url = new URL(a.getAttribute("href"), location.href);
        if (url.pathname.slice(-STORY.length) !== STORY || !url.hash) return;

        e.preventDefault();
        var id = decodeURIComponent(url.hash.slice(1));
        loadStory()
            .then(function (doc) {
                var entry = doc.getElementById(id);
                if (entry) show(a, entry, url.href);
                else location.href = url.href;
            })
            .catch(function () {
                location.href = url.href;
            });
    });

    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") close();
    });
})();
