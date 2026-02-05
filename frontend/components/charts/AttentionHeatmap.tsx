"use client";
import { useMemo } from "react";

interface AttentionHeatmapProps {
  weights: number[][];
  labels?: string[];
}

export default function AttentionHeatmap({ weights, labels }: AttentionHeatmapProps) {
  const { cells, maxVal } = useMemo(() => {
    let max = 0;
    const cells: { row: number; col: number; value: number }[] = [];
    for (let r = 0; r < weights.length; r++) {
      for (let c = 0; c < weights[r].length; c++) {
        const v = weights[r][c];
        if (v > max) max = v;
        cells.push({ row: r, col: c, value: v });
      }
    }
    return { cells, maxVal: max };
  }, [weights]);

  const size = weights.length;
  const cellSize = Math.min(40, 400 / size);

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-4">
        Attention Weights
      </h3>
      <div className="overflow-x-auto">
        <svg
          width={cellSize * size + 60}
          height={cellSize * size + 40}
          className="mx-auto"
        >
          {/* Row labels */}
          {Array.from({ length: size }, (_, i) => (
            <text
              key={`row-${i}`}
              x={50}
              y={20 + i * cellSize + cellSize / 2 + 4}
              textAnchor="end"
              fill="#8899aa"
              fontSize={10}
            >
              {labels ? labels[i] : `T${i + 1}`}
            </text>
          ))}
          {/* Column labels */}
          {Array.from({ length: size }, (_, i) => (
            <text
              key={`col-${i}`}
              x={60 + i * cellSize + cellSize / 2}
              y={12}
              textAnchor="middle"
              fill="#8899aa"
              fontSize={10}
            >
              {labels ? labels[i] : `T${i + 1}`}
            </text>
          ))}
          {/* Cells */}
          {cells.map(({ row, col, value }) => {
            const intensity = maxVal > 0 ? value / maxVal : 0;
            const r = Math.round(230 * intensity);
            const g = Math.round(57 * intensity);
            const b = Math.round(70 * intensity);
            return (
              <g key={`${row}-${col}`}>
                <rect
                  x={60 + col * cellSize}
                  y={20 + row * cellSize}
                  width={cellSize - 1}
                  height={cellSize - 1}
                  fill={`rgb(${r}, ${g}, ${b})`}
                  rx={2}
                />
                {cellSize >= 28 && (
                  <text
                    x={60 + col * cellSize + cellSize / 2}
                    y={20 + row * cellSize + cellSize / 2 + 3}
                    textAnchor="middle"
                    fill={intensity > 0.5 ? "#fff" : "#8899aa"}
                    fontSize={9}
                  >
                    {value.toFixed(2)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
