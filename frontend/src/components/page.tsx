"use client";

import React, { useState, useEffect } from "react";
import {
  AppstoreOutlined,
  NodeIndexOutlined,
  ClusterOutlined,
  TableOutlined,
  PlayCircleOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  UserOutlined,
  BranchesOutlined,
  CodeOutlined,
  DeploymentUnitOutlined,
  SearchOutlined,
  DatabaseOutlined,
  PieChartOutlined,
  BulbOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import {
  Layout,
  Menu,
  Switch,
  Typography,
  ConfigProvider,
  theme as antdTheme,
  Avatar,
  Tooltip,
} from "antd";
import App from "./AnalyticsDashboardComponent";
import { BoundaryEvaluationComponent } from "./BoundaryEvaluationComponent";
import { BuildKGFromTripletsComponent } from "./BuildKGFromTripletsComponent";
import { DetectSentenceBoundaryComponent } from "./DetectSentenceBoundaryComponent";
import { EntropyTraversalComponent } from "./EntropyTraversalComponent";
import { EvaluateBoundariesNormalizedComponent } from "./EvaluateBoundariesNormalizedComponent";
import { ExtractSVOComponent } from "./ExtractSVOComponent";
import { GenerateNodesToStartComponent } from "./GenerateNodesToStartComponent";
import { GNNBoundaryExperimentComponent } from "./GNNBoundaryExperimentComponent";
import { KGVisualizationComponent } from "./KGVisualizationComponent";
import { NebulaGraphUploadAndVisualizeComponent } from "./NebulaGraphUploadAndVisualizeComponent";
import { Neo4jGraphComponent } from "./Neo4jGraphComponent";
import { Node2VecEmbeddingsComponent } from "./Node2VecEmbeddingsComponent";
import PatchEmbeddingApp from "./PatchEmbedding";
import { SelectStartingNodesComponent } from "./SelectStartingNodesComponent";

const logoUrl = "/logo192.png";

const { Title } = Typography;
const { Sider, Content, Header } = Layout;

const MENU = [
  // Sequence follows backend process
  {
    key: "generate-nodes",
    icon: <BranchesOutlined />,
    label: "1. Generate Starting Nodes",
    description: "Extract main subjects as starting points for traversal",
    component: <GenerateNodesToStartComponent />,
  },
  {
    key: "extract-svo",
    icon: <TableOutlined />,
    label: "2. Extract SVO Triplets",
    description: "Extract Subject-Verb-Object triplets from text",
    component: <ExtractSVOComponent />,
  },
  {
    key: "build-kg",
    icon: <ClusterOutlined />,
    label: "3. Build Knowledge Graph",
    description: "Build a knowledge graph from triplets",
    component: <BuildKGFromTripletsComponent />,
  },
  {
    key: "visualize-kg",
    icon: <PieChartOutlined />,
    label: "4. Visualize Knowledge Graph",
    description: "Visualize entities and relationships",
    component: <KGVisualizationComponent />,
  },
  {
    key: "node2vec-embeddings",
    icon: <DeploymentUnitOutlined />,
    label: "5. Node2Vec Embeddings",
    description: "Generate node2vec embeddings",
    component: <Node2VecEmbeddingsComponent />,
  },
  {
    key: "select-nodes",
    icon: <BulbOutlined />,
    label: "6. Select Starting Nodes",
    description: "Select nodes for boundary detection",
    component: <SelectStartingNodesComponent />,
  },
  {
    key: "entropy-traversal",
    icon: <PlayCircleOutlined />,
    label: "7. Entropy Boundary Detection",
    description: "Detect boundaries using entropy traversal",
    component: <EntropyTraversalComponent />,
  },
  {
    key: "detect-boundaries",
    icon: <SearchOutlined />,
    label: "8. Detect Sentence Boundaries",
    description: "Detect boundaries and post-process",
    component: <DetectSentenceBoundaryComponent />,
  },
  {
    key: "evaluate-boundaries",
    icon: <BarChartOutlined />,
    label: "9. Evaluate Boundaries",
    description: "Evaluate predictions (raw/normalized)",
    component: <BoundaryEvaluationComponent />,
  },
  {
    key: "evaluate-boundaries-normalized",
    icon: <ExperimentOutlined />,
    label: "10. Evaluate Normalized Boundaries",
    description: "Evaluate with normalization",
    component: <EvaluateBoundariesNormalizedComponent />,
  },
  {
    key: "gnn-boundary-experiment",
    icon: <CodeOutlined />,
    label: "11. GNN Boundary Classifier",
    description: "GNN-based classifier and experiments",
    component: <GNNBoundaryExperimentComponent />,
  },
  {
    key: "patch-embedding",
    icon: <AppstoreOutlined />,
    label: "12. Patch Embedding Training",
    description: "Train patch embedding autoencoder",
    component: <PatchEmbeddingApp />,
  },
  {
    key: "analytics-dashboard",
    icon: <BarChartOutlined />,
    label: "13. Analytics Dashboard",
    description: "Interactive patch and KG analytics",
    component: <App />,
  },
  {
    key: "neo4j-graph",
    icon: <DatabaseOutlined />,
    label: "14. Neo4j Graph Demo",
    description: "Neo4j graph ingestion/visualization",
    component: <Neo4jGraphComponent />,
  },
  {
    key: "nebula-graph",
    icon: <DatabaseOutlined />,
    label: "15. Nebula Graph Demo",
    description: "Nebula graph ingestion/visualization",
    component: <NebulaGraphUploadAndVisualizeComponent />,
  },
];

export default function MainPage() {
  // Theme: dark/light
  const [darkMode, setDarkMode] = useState<boolean>(false);

  // Selected menu key (initially first in sequence)
  const [selectedKey, setSelectedKey] = useState(MENU[0].key);

  // Menu for the sidebar
  const menuItems = MENU.map(({ key, icon, label, description }) => ({
    key,
    icon,
    label: (
      <Tooltip title={description} placement="right">
        <span>{label}</span>
      </Tooltip>
    ),
  }));

  // Find active component to render
  const activeComponent = MENU.find(
    (item) => item.key === selectedKey
  )?.component;

  // Responsive: collapse sidebar on mobile
  const [collapsed, setCollapsed] = useState(false);

  // Set page background for theme
  useEffect(() => {
    document.body.style.background = darkMode ? "#10141a" : "#f4f6fa";
  }, [darkMode]);

  return (
    <ConfigProvider
      theme={{
        algorithm: darkMode
          ? antdTheme.darkAlgorithm
          : antdTheme.defaultAlgorithm,
        token: {
          colorPrimary: "#1890ff",
          borderRadius: 12,
        },
      }}
    >
      <Layout style={{ minHeight: "100vh", transition: "background 0.2s" }}>
        {/* Sidebar */}
        <Sider
          width={240}
          collapsible
          collapsed={collapsed}
          onCollapse={setCollapsed}
          theme={darkMode ? "dark" : "light"}
          style={{
            boxShadow: "2px 0 8px #0001",
            background: darkMode ? "#161a23" : "#fff",
            transition: "background 0.2s",
          }}
        >
          <div
            style={{
              margin: "24px auto 16px",
              textAlign: "center",
            }}
          >
            <img
              src={logoUrl}
              alt="Logo"
              style={{
                width: collapsed ? 32 : 56,
                height: collapsed ? 32 : 56,
                borderRadius: "50%",
                background: darkMode ? "#24292f" : "#f4f6fa",
                boxShadow: "0 2px 8px #0002",
                marginBottom: 8,
                transition: "all 0.2s",
              }}
            />
            {!collapsed && (
              <Title
                level={4}
                style={{
                  margin: 0,
                  color: darkMode ? "#fff" : "#222",
                  fontWeight: 700,
                  letterSpacing: 1,
                }}
              >
                KG Pipeline
              </Title>
            )}
          </div>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            items={menuItems}
            onClick={({ key }) => setSelectedKey(key as string)}
            style={{
              borderRight: 0,
              marginTop: 12,
              fontSize: 16,
            }}
            theme={darkMode ? "dark" : "light"}
          />
          <div
            style={{
              position: "absolute",
              bottom: 0,
              width: "100%",
              textAlign: "center",
              padding: collapsed ? "10px 0" : "16px 0",
              background: darkMode ? "#181c24" : "#f9fafb",
              borderTop: darkMode ? "1px solid #232733" : "1px solid #e8eaf1",
              transition: "background 0.2s",
            }}
          >
            <Switch
              checkedChildren="🌙"
              unCheckedChildren="☀️"
              checked={darkMode}
              onChange={setDarkMode}
              style={{ marginRight: 8 }}
            />
            {!collapsed && (
              <span style={{ color: darkMode ? "#ccc" : "#666", fontSize: 13 }}>
                Dark Mode
              </span>
            )}
            <div style={{ marginTop: collapsed ? 0 : 14, textAlign: "center" }}>
              <Avatar
                src="https://avatars.githubusercontent.com/u/3369400?v=4"
                size={collapsed ? 28 : 40}
              />
              {!collapsed && (
                <div
                  style={{
                    fontSize: 13,
                    color: darkMode ? "#ccc" : "#222",
                    marginTop: 6,
                  }}
                >
                  <b>sirjanhere</b>
                </div>
              )}
            </div>
          </div>
        </Sider>
        {/* Main Content */}
        <Layout>
          <Header
            style={{
              background: darkMode ? "#1a1e27" : "#fff",
              padding: "0 32px",
              borderBottom: darkMode
                ? "1px solid #232733"
                : "1px solid #e8eaf1",
              display: "flex",
              alignItems: "center",
              minHeight: 60,
              boxShadow: "0 1px 8px #0001",
            }}
          >
            <Title
              level={3}
              style={{
                color: darkMode ? "#fff" : "#181a20",
                margin: 0,
                fontWeight: 700,
                letterSpacing: 1,
                fontSize: 22,
              }}
            >
              {MENU.find((m) => m.key === selectedKey)?.label}
            </Title>
          </Header>
          <Content
            style={{
              background: darkMode ? "#12151c" : "#f7fafd",
              padding: 40,
              minHeight: "calc(100vh - 60px)",
              overflow: "auto",
            }}
          >
            <div
              style={{
                background: darkMode ? "#181c24" : "#fff",
                borderRadius: 18,
                boxShadow: darkMode ? "0 2px 16px #0003" : "0 2px 16px #e0e5f6",
                padding: 32,
                minHeight: 400,
                maxWidth: 1200,
                margin: "0 auto",
              }}
            >
              {activeComponent}
            </div>
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}
