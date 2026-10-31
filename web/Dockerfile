FROM node:22-alpine AS dependencies

WORKDIR /app
COPY web/package.json web/package-lock.json ./
RUN npm ci

FROM node:22-alpine AS runner

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY web ./
RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
