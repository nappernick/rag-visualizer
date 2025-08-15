import React, { useState, useEffect } from 'react';
import { Chunk } from '../../types';

interface HierarchyTreeProps {
  chunks: Chunk[];
  selectedChunkId?: string;
  onChunkSelect: (chunkId: string) => void;
}

interface TreeNode {
  chunk: Chunk;
  children: TreeNode[];
  expanded: boolean;
}

export const HierarchyTree: React.FC<HierarchyTreeProps> = ({
  chunks,
  selectedChunkId,
  onChunkSelect
}) => {
  const [treeNodes, setTreeNodes] = useState<TreeNode[]>([]);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Build tree structure from chunks
    const buildTree = () => {
      const nodeMap = new Map<string, TreeNode>();
      const rootNodes: TreeNode[] = [];

      // First pass: create all nodes
      chunks.forEach(chunk => {
        nodeMap.set(chunk.id, {
          chunk,
          children: [],
          expanded: expandedNodes.has(chunk.id) || chunk.level === 0
        });
      });

      // Second pass: build parent-child relationships
      chunks.forEach(chunk => {
        const node = nodeMap.get(chunk.id);
        if (!node) return;

        if (chunk.parent_id) {
          const parentNode = nodeMap.get(chunk.parent_id);
          if (parentNode) {
            parentNode.children.push(node);
          } else {
            // No parent found, treat as root
            rootNodes.push(node);
          }
        } else {
          // No parent, it's a root node
          rootNodes.push(node);
        }
      });

      // Sort children by chunk_index
      const sortNodes = (nodes: TreeNode[]) => {
        nodes.sort((a, b) => a.chunk.chunk_index - b.chunk.chunk_index);
        nodes.forEach(node => sortNodes(node.children));
      };
      sortNodes(rootNodes);

      return rootNodes;
    };

    setTreeNodes(buildTree());
  }, [chunks, expandedNodes]);

  const toggleExpand = (chunkId: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(chunkId)) {
        newSet.delete(chunkId);
      } else {
        newSet.add(chunkId);
      }
      return newSet;
    });
  };

  const renderTreeNode = (node: TreeNode, depth: number = 0): JSX.Element => {
    const isSelected = selectedChunkId === node.chunk.id;
    const hasChildren = node.children.length > 0;
    const isExpanded = expandedNodes.has(node.chunk.id);
    const level = node.chunk.level ?? depth;

    // Get chunk type from metadata
    const chunkType = node.chunk.metadata?.type || 'content';
    const title = node.chunk.metadata?.title || `Chunk ${node.chunk.chunk_index}`;

    // Determine icon based on type
    const getIcon = () => {
      if (chunkType === 'section_header') return '📑';
      if (chunkType === 'group_parent') return '📁';
      if (chunkType === 'section_content') return '📄';
      if (level === 0) return '📚';
      if (level === 1) return '📂';
      if (level === 2) return '📝';
      return '📃';
    };

    // Get level-based styling
    const getLevelStyles = () => {
      const baseClasses = "flex items-center py-2 px-3 rounded-lg cursor-pointer transition-all duration-200";
      const levelClasses = [
        "bg-gradient-to-r from-purple-50 to-indigo-50 hover:from-purple-100 hover:to-indigo-100 font-semibold",
        "bg-gradient-to-r from-blue-50 to-cyan-50 hover:from-blue-100 hover:to-cyan-100",
        "bg-gradient-to-r from-green-50 to-emerald-50 hover:from-green-100 hover:to-emerald-100",
        "bg-gray-50 hover:bg-gray-100"
      ];
      
      const selectedClass = isSelected 
        ? "ring-2 ring-blue-500 shadow-lg scale-102" 
        : "";
      
      return `${baseClasses} ${levelClasses[Math.min(level, 3)]} ${selectedClass}`;
    };

    return (
      <div key={node.chunk.id} className="mb-1">
        <div
          style={{ paddingLeft: `${depth * 20}px` }}
          className={getLevelStyles()}
          onClick={() => onChunkSelect(node.chunk.id)}
        >
          {/* Expand/Collapse button */}
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpand(node.chunk.id);
              }}
              className="mr-2 text-gray-500 hover:text-gray-700 transition-colors"
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          )}
          {!hasChildren && <span className="mr-6" />}

          {/* Icon */}
          <span className="mr-2 text-lg">{getIcon()}</span>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm truncate">
                {title}
              </span>
              <div className="flex items-center gap-2 ml-2">
                {/* Token count */}
                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                  {node.chunk.tokens} tokens
                </span>
                {/* Child count */}
                {hasChildren && (
                  <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                    {node.children.length} {node.children.length === 1 ? 'child' : 'children'}
                  </span>
                )}
                {/* Level indicator */}
                <span className="text-xs text-purple-600 bg-purple-100 px-2 py-1 rounded">
                  L{level}
                </span>
              </div>
            </div>
            
            {/* Preview of content */}
            <p className="text-xs text-gray-600 mt-1 truncate">
              {node.chunk.content.substring(0, 100)}...
            </p>
          </div>
        </div>

        {/* Render children if expanded */}
        {hasChildren && isExpanded && (
          <div className="mt-1">
            {node.children.map(child => renderTreeNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  // Calculate statistics
  const stats = {
    totalChunks: chunks.length,
    maxDepth: Math.max(...chunks.map(c => c.level ?? 0)),
    rootNodes: treeNodes.length,
    hierarchical: chunks.some(c => c.parent_id || (c.children_ids && c.children_ids.length > 0))
  };

  return (
    <div className="h-full flex flex-col">
      {/* Statistics Bar */}
      <div className="mb-4 p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-200">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-sm text-gray-700">
            {stats.hierarchical ? '🌳 Hierarchical Structure' : '📝 Linear Structure'}
          </h3>
          <div className="flex gap-3 text-xs">
            <span className="text-gray-600">
              Total: <span className="font-bold text-indigo-600">{stats.totalChunks}</span>
            </span>
            <span className="text-gray-600">
              Depth: <span className="font-bold text-purple-600">{stats.maxDepth + 1}</span>
            </span>
            <span className="text-gray-600">
              Roots: <span className="font-bold text-blue-600">{stats.rootNodes}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Tree Controls */}
      <div className="mb-3 flex gap-2">
        <button
          onClick={() => setExpandedNodes(new Set(chunks.map(c => c.id)))}
          className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
        >
          Expand All
        </button>
        <button
          onClick={() => setExpandedNodes(new Set())}
          className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
        >
          Collapse All
        </button>
      </div>

      {/* Tree View */}
      <div className="flex-1 overflow-y-auto bg-white rounded-lg border border-gray-200 p-4">
        {treeNodes.length > 0 ? (
          treeNodes.map(node => renderTreeNode(node, 0))
        ) : (
          <div className="text-center text-gray-500 py-8">
            <span className="text-3xl">🌲</span>
            <p className="mt-2">No hierarchical structure detected</p>
            <p className="text-sm mt-1">Chunks are displayed in linear order</p>
          </div>
        )}
      </div>

      {/* Selected Chunk Preview */}
      {selectedChunkId && (
        <div className="mt-4 p-4 bg-gradient-to-r from-gray-50 to-slate-50 rounded-lg border border-gray-200">
          <h4 className="font-semibold text-sm mb-2">Selected Chunk Content</h4>
          <pre className="text-xs whitespace-pre-wrap max-h-32 overflow-y-auto bg-white p-3 rounded border border-gray-200">
            {chunks.find(c => c.id === selectedChunkId)?.content || 'Chunk not found'}
          </pre>
        </div>
      )}
    </div>
  );
};