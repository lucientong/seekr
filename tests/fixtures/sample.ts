// Test fixture: TypeScript sample for AST golden / structural gates.

export async function fetchUserProfile(userId: number): Promise<UserProfile> {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch user: ${response.status}`);
  }
  return response.json();
}

export function formatCurrency(amount: number, currency: string): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(
    amount,
  );
}

export class TokenBucket {
  private tokens: number;

  constructor(capacity: number) {
    this.tokens = capacity;
  }

  tryAcquire(cost: number): boolean {
    if (this.tokens < cost) {
      return false;
    }
    this.tokens -= cost;
    return true;
  }
}

export interface UserProfile {
  id: number;
  name: string;
  email: string;
}

export type AuthResult = { ok: true; token: string } | { ok: false; reason: string };
