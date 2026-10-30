import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { Nav } from "@/components/site/Nav";
import { Workspace } from "@/components/workspace/Workspace";
import { listRunIds, loadRun } from "@/lib/bundle";

export const dynamicParams = false;

export async function generateStaticParams() {
  return (await listRunIds()).map(({ id }) => ({ id }));
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const run = await loadRun(id);
  if (!run) return { title: "Run not found" };
  return {
    title: run.protocol?.title ?? run.title,
    description: run.question ?? undefined,
    robots: run.origin === "dev" ? { index: false } : undefined,
  };
}

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const run = await loadRun(id);
  if (!run) notFound();
  return (
    <>
      <Nav current="/runs" />
      <main>
        <Workspace initial={run} />
      </main>
    </>
  );
}
