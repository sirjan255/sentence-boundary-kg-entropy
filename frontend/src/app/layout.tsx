export const metadata = {
  title: "Sentence Boundary KG Analytics",
  description: "Sentence Boundary KG Analytics",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ margin: "0" }}>{children}</body>
    </html>
  );
}
