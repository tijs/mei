document.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-copy]');
  if (button) {
    const code = button.closest('.code-block').querySelector('code').textContent;
    try {
      await navigator.clipboard.writeText(code);
      button.textContent = 'Copied';
      document.querySelector('#copy-status').textContent = 'Code copied to clipboard.';
      setTimeout(() => { button.textContent = 'Copy'; }, 1800);
    } catch {
      document.querySelector('#copy-status').textContent = 'Copy unavailable. Select the code to copy it.';
    }
  }
  if (event.target.closest('.mobile-menu a')) document.querySelector('.mobile-menu').open = false;
});
document.addEventListener('htmx:responseError', () => {
  // The links always work as full static pages even if a fragment request fails.
  const panel = document.querySelector('#results-panel');
  if (!panel.querySelector('.chart-error')) {
    const message = document.createElement('p');
    message.className = 'chart-error';
    message.textContent = 'Couldn’t load this view. Open the measurement link in a new tab to view the full page.';
    panel.append(message);
  }
});
