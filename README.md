# MIT Campus Decarbonization
Materials for MIT Campus Decarbonization Study

![Technology Overview](./figures/technology-overview.drawio.png)

## Setup

1. Install `npm`, `conda`, `docker`
1. `cp .env.example .env`
1. `npm i`
1. `npm run conda-setup`
1. `npm run supa-login` (you will need to login on the browser, then select the appropriate project)
1. `npx supabase start`
1. `npm run db-refresh`

## Running the Data Warehouse

First, reset the db to a clean state, applying all migrations present and then seeding with data.

```sh
npm run db-refresh
```

Start the streamlit frontend:

```sh
npm run fe-start
```

## Database Migration

Database migration is managed with supabase.

(coming soon)


