import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    testTimeout: 5 * 60 * 1000,
    hookTimeout: 60 * 1000,
    // Model-backed tests load real weights sequentially; running them
    // concurrently would multiply peak memory well past what CI runners
    // (or a laptop) can hold at once.
    fileParallelism: false,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["index.js"],
    },
  },
});
