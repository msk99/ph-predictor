// Service Worker for pH Predictor PWA
const CACHE_NAME = 'ph-predictor-v5';
const MODEL_CACHE = 'ph-model-v1';

// App shell files (cached immediately on install)
const APP_SHELL = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
  './help.html',
  './example_pH7.jpg'
];

// Model file (cached on first use)
const MODEL_URL = './model/model.tflite';

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME && k !== MODEL_CACHE)
          .map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Model file — cache on first fetch, serve from cache thereafter
  if (url.pathname.endsWith('.tflite')) {
    event.respondWith(
      caches.open(MODEL_CACHE).then(cache =>
        cache.match(event.request).then(cached => {
          if (cached) return cached;
          return fetch(event.request).then(response => {
            cache.put(event.request, response.clone());
            return response;
          });
        })
      )
    );
    return;
  }

  // App shell — cache first, network fallback
  event.respondWith(
    caches.match(event.request)
      .then(cached => cached || fetch(event.request))
  );
});
