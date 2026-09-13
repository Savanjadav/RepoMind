export class RetryPolicy {
  maxAttempts = 3;
}

export function retryFailedJob(attempt: number, maxAttempts: number): boolean {
  return attempt < maxAttempts;
}
