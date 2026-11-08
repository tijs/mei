document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy]");
  if (button) {
    const code = button
      .closest(".code-block")
      .querySelector("code").textContent;
    try {
      await navigator.clipboard.writeText(code);
      button.textContent = "Copied";
      document.querySelector("#copy-status").textContent =
        "Code copied to clipboard.";
      setTimeout(() => {
        button.textContent = "Copy";
      }, 1800);
    } catch {
      document.querySelector("#copy-status").textContent =
        "Couldn’t copy. Select the code and copy it manually.";
    }
  }
  if (event.target.closest(".mobile-menu a"))
    document.querySelector(".mobile-menu").open = false;
});
document.addEventListener("htmx:responseError", () => {
  // The links always work as full static pages even if a fragment request fails.
  const panel = document.querySelector("#results-panel");
  if (!panel.querySelector(".chart-error")) {
    const message = document.createElement("p");
    message.className = "chart-error";
    message.textContent =
      "The chart couldn’t load. Open the link for that chart in a new browser tab to load the full page.";
    panel.append(message);
  }
});
