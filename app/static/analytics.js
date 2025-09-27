// Lightweight analytics bootstrap (no-op)
(function () {
  try {
    const eventQueue = [];
    function track(eventName, payload) {
      eventQueue.push({ eventName, payload: payload || {}, ts: Date.now() });
    }
    // Expose a tiny API without leaking globals badly
    window.MediTrackAnalytics = { track, _queue: eventQueue };

    // Basic page view
    track('page_view', { path: window.location.pathname });

    // Optional: track clicks on elements with data-track attribute
    document.addEventListener('click', function (e) {
      const el = e.target.closest('[data-track]');
      if (!el) return;
      const name = el.getAttribute('data-track') || 'click';
      track(name, { id: el.id || null, text: (el.textContent || '').trim().slice(0, 64) });
    }, { capture: true });

    // Debug log in dev
    if (typeof process === 'undefined' || (process && process.env && process.env.NODE_ENV !== 'production')) {
      // eslint-disable-next-line no-console
      console.debug('[MediTrack] analytics loaded');
    }
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[MediTrack] analytics disabled:', e && e.message ? e.message : e);
  }
})();


