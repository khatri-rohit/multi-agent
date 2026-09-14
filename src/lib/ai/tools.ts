import { tool } from 'langchain';
import { z } from 'zod';
import { getFirecrawl } from '../firecrawl';

// This tool is used to get the search urls for the given input
export const getSearchResults = tool(
    async (input: string) => {
        const firecrawl = getFirecrawl();

        const results = await firecrawl.search(input, {
            limit: 3,
            scrapeOptions: { formats: ['markdown'] },
        });
        console.log('Search results: ', results);

        return results.web;
    },
    {
        name: 'get_search_urls',
        description: 'Use this tool to get the search urls for the given input',
        schema: z.object({
            input: z.string(),
        }),
    },
);

// This tool is used to scrape the websites for the given urls
export const scrapeWebsite = tool(
    async (urls: string[]) => {
        const firecrawl = getFirecrawl();

        const { id } = await firecrawl.batchScrape(urls, {
            options: {
                formats: ['markdown'],
            },
        });

        const job = await firecrawl.getBatchScrapeStatus(id);
        console.log(
            'Scrape job status: ',
            job.status,
            job.completed,
            job.total,
        );

        return job.data;
    },
    {
        name: 'scrape_websites',
        description: 'Use this tool to scrape the websites for the given urls',
        schema: z.object({
            urls: z.array(z.string()),
        }),
    },
);
