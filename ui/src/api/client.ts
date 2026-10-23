// Typed API client: the types come from the FastAPI schema (npm run gen:api), so the browser and the tables can't drift.
import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export const api = createClient<paths>({ baseUrl: "" });
export type S = components["schemas"];

export async function get<T>(p: Promise<{ data?: T; error?: unknown }>): Promise<T> {
  const r = await p;
  if (r.error || r.data === undefined) throw new Error(JSON.stringify(r.error ?? "no data"));
  return r.data;
}
