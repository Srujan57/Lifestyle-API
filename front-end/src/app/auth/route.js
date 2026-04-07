import { NextResponse } from "next/server";

export async function GET() {
  const API_BASE_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
  return NextResponse.redirect(new URL(`${API_BASE_URL}/authorize`));
}
