import { tool } from 'langchain';

import { z } from 'zod';

import { getFirecrawl } from '../firecrawl';

function toolErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

// Tool to get search results from the web
export const getSearchResults = tool(
    async ({ input }: { input: string }) => {
        const query = input.trim();
        if (!query) {
            return { error: 'Search query must be a non-empty string.' };
        }
        console.log('Getting search results for:', query);

        try {
            const firecrawl = getFirecrawl();
            const results = await firecrawl.search(query, {
                limit: 3,
                scrapeOptions: { formats: ['markdown'] },
            });
            console.log('Search results count:', results.web?.length ?? 0);

            return {
                web: results.web ?? [],
                news: results.news ?? [],
            };
        } catch (error) {
            console.error('get_search_urls failed:', error);

            return { error: toolErrorMessage(error) };
        }
    },
    {
        name: 'get_search_urls',
        description:
            'Search the web for URLs and snippets. Pass a single search query string as `input`.',
        schema: z.object({
            input: z
                .string()
                .describe('Search query, e.g. "Bleach TYBW IMDb rating"'),
        }),
    },
);

// Tool to scrape websites
export const scrapeWebsite = tool(
    async ({ urls }: { urls: string[] }) => {
        const list = urls.map((u) => u.trim()).filter(Boolean);
        if (!list.length) {
            return { error: 'At least one URL is required in `urls`.' };
        }
        console.log('Scraping websites for:', list);
        try {
            const firecrawl = getFirecrawl();
            const job = await firecrawl.batchScrape(list, {
                options: { formats: ['markdown'] },
                pollInterval: 2,
                timeout: 120,
            });
            console.log(
                'Scrape job status:',
                job.status,
                job.completed,
                job.total,
            );

            return {
                status: job.status,
                completed: job.completed,
                total: job.total,
                data: job.data ?? [],
            };
        } catch (error) {
            console.error('scrape_websites failed:', error);
            return { error: toolErrorMessage(error) };
        }
    },
    {
        name: 'scrape_websites',
        description:
            'Scrape one or more URLs and return markdown content. Pass `urls` as a string array.',
        schema: z.object({
            urls: z
                .array(z.string().url())
                .min(1)
                .describe('Full URLs to scrape, e.g. ["https://example.com"]'),
        }),
    },
);

// Tool to write a summary of the information provided to you
export const writeSummary = tool(
    async ({ information }: { information: string }) => {
        console.log('Writing summary for:', information);
        return {
            summary: `Here is the summary of the information provided to you. ${information}`,
        };
    },
    {
        name: 'write_summary',
        description:
            'Write a summary of the information provided to you. Pass `input` as a string.',
        schema: z.object({
            information: z
                .string()
                .describe('Information to write a summary of.'),
        }),
    },
);
