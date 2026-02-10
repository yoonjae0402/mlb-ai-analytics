interface ChartExplainerProps {
  children: React.ReactNode;
}

export default function ChartExplainer({ children }: ChartExplainerProps) {
  return (
    <p className="text-xs text-mlb-muted/80 leading-relaxed mb-3">
      {children}
    </p>
  );
}
