import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { Nav } from "@/components/site/Nav";
import { LiveRun } from "@/components/workspace/LiveRun";

export const metadata: Metadata = { title: "Live run", robots: { index: false } };

export default async function LivePage({ params }: { params: Promise<{ id: string }> }) {
  if (!process.env.NEXT_PUBLIC_KAPPA_API) notFound();
  const { id } = await params;
  return (
    <>
      <Nav current="/runs" />
      <main>
        <LiveRun id={id} />
      </main>
    </>
  );
}
