import { Firecrawl } from 'firecrawl';

export const getFirecrawl = () => {
    return new Firecrawl({
        apiKey: process.env.FIRECRAWL_API_KEY,
        maxRetries: 3,
    });
};
