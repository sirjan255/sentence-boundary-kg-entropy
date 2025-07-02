import React from "react";

export function Card({
  children,
  title,
  bordered = true,
  style,
  ...props
}: React.PropsWithChildren<{ title?: React.ReactNode; bordered?: boolean; style?: React.CSSProperties }>) {
  return (
    <div
      style={{
        border: bordered ? "1px solid #e5e7eb" : "none",
        borderRadius: 8,
        boxShadow: bordered ? "0 1px 4px rgba(0,0,0,0.05)" : undefined,
        padding: 20,
        background: "#fff",
        marginBottom: 16,
        ...style,
      }}
      {...props}
    >
      {title && (
        <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 12 }}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export function CardHeader({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }} {...props}>{children}</div>;
}

export function CardTitle({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div style={{ fontWeight: 500, fontSize: 14 }} {...props}>{children}</div>;
}

export function CardContent({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div {...props}>{children}</div>;
}

export function CardDescription({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div style={{ color: "#64748b", fontSize: 12, marginTop: 2 }} {...props}>{children}</div>;
}