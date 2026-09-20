# Vercel deployment

Import this repository into Vercel and set **Root Directory** to `web`. The checked-in `vercel.json` runs the native Next.js production build.

## Required Vercel environment variables

- `NEXT_PUBLIC_SHAPER_API_URL`: the public HTTPS origin of the GPU backend, without a trailing slash. This direct connection is required because reconstruction progress and phone pairing use WebSockets, which cannot be proxied by a Vercel Function.
- `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY`: the browser-restricted Google key with **Maps JavaScript API** enabled. Add `https://*.vercel.app/*` plus the production custom domain to its Website restrictions.

Optional variables are documented in `.env.example`. Configure them for Production and Preview environments before deploying.

## Backend environment

The backend needs:

- `GOOGLE_MAPS_API_KEY`: server-restricted key with **Routes API** enabled.
- `GOOGLE_MAPS_BROWSER_API_KEY`: key with **Places API (New)** enabled for address suggestions. It can be the same value used for `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY`.
- `SHAPER_CORS_ORIGINS`: comma-separated exact frontend origins. The default already includes `https://honkpack.vercel.app`; add any future custom domain here.
- `SHAPER_CORS_ORIGIN_REGEX` (optional): override only when you control the matching preview domains; prefer exact entries in `SHAPER_CORS_ORIGINS`.

Never commit real keys. After changing any `NEXT_PUBLIC_*` value, redeploy because Next.js embeds it at build time.
