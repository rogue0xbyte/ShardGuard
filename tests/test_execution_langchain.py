"""Tests for the execution module."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.tools import Tool
from langchain.agents import AgentType

from shardguard.core.execution_langchain import (
    GenericExecutionLLMWrapper,
    ToolsWrapper,
    make_langchain_tools,
    make_execution_agent,
    EXEC_SYSTEM_PROMPT,
)


class TestGenericExecutionLLMWrapper:
    """Test the GenericExecutionLLMWrapper function."""

    @pytest.fixture
    def mock_generic_llm(self):
        """Create a mock GenericExecutionLLM."""
        mock_llm = Mock()
        mock_llm.provider_type = "ollama"
        mock_llm.model = "llama3.2"
        mock_llm.llm_provider = Mock()
        return mock_llm

    def test_wrapper_creation(self, mock_generic_llm):
        """Test that wrapper LLM is created successfully."""
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        assert wrapper is not None
        assert wrapper._llm_type == "generic_execution_llm"

    def test_wrapper_identifying_params(self, mock_generic_llm):
        """Test that identifying params are correctly exposed."""
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        params = wrapper._identifying_params
        assert params["provider_type"] == "ollama"
        assert params["model"] == "llama3.2"

    def test_wrapper_call_raises_not_implemented(self, mock_generic_llm):
        """Test that synchronous _call raises NotImplementedError."""
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        with pytest.raises(NotImplementedError, match="Use async _acall"):
            wrapper._call("test prompt")

    @pytest.mark.asyncio
    async def test_wrapper_acall_success_string_response(self, mock_generic_llm):
        """Test async call with string response."""
        mock_generic_llm.llm_provider.generate_response = AsyncMock(
            return_value="Test response"
        )
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        result = await wrapper._acall("test prompt")
        
        assert result == "Test response"
        mock_generic_llm.llm_provider.generate_response.assert_called_once_with(
            "test prompt"
        )

    @pytest.mark.asyncio
    async def test_wrapper_acall_success_dict_response(self, mock_generic_llm):
        """Test async call with dict response converts to JSON."""
        response_dict = {"key": "value", "number": 42}
        mock_generic_llm.llm_provider.generate_response = AsyncMock(
            return_value=response_dict
        )
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        result = await wrapper._acall("test prompt")
        
        assert result == json.dumps(response_dict)
        mock_generic_llm.llm_provider.generate_response.assert_called_once_with(
            "test prompt"
        )

    @pytest.mark.asyncio
    async def test_wrapper_acall_with_stop_parameter(self, mock_generic_llm):
        """Test async call with stop parameter."""
        mock_generic_llm.llm_provider.generate_response = AsyncMock(
            return_value="Response"
        )
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        result = await wrapper._acall("test prompt", stop=["STOP", "END"])
        
        assert result == "Response"

    @pytest.mark.asyncio
    async def test_wrapper_acall_error_handling(self, mock_generic_llm):
        """Test async call error handling."""
        mock_generic_llm.llm_provider.generate_response = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        result = await wrapper._acall("test prompt")
        
        assert result == "[ERROR]: Connection failed"


class TestToolsWrapper:
    """Test the ToolsWrapper function."""

    def test_tools_wrapper_single_tool(self):
        """Test wrapping a single tool with server prefix."""
        tools_list = ["file-server.read-file"]
        
        result = ToolsWrapper(tools_list)
        
        assert len(result) == 1
        assert isinstance(result[0], Tool)
        assert result[0].name == "read-file"
        assert result[0].description == ""

    def test_tools_wrapper_multiple_tools(self):
        """Test wrapping multiple tools."""
        tools_list = [
            "file-server.read-file",
            "file-server.write-file",
            "database-server.query"
        ]
        
        result = ToolsWrapper(tools_list)
        
        assert len(result) == 3
        assert result[0].name == "read-file"
        assert result[1].name == "write-file"
        assert result[2].name == "query"

    def test_tools_wrapper_tool_without_server_prefix(self):
        """Test wrapping tool without server prefix uses default."""
        tools_list = ["standalone-tool"]
        
        result = ToolsWrapper(tools_list)
        
        assert len(result) == 1
        assert result[0].name == "standalone-tool"

    def test_tools_wrapper_tool_function_returns_dict(self):
        """Test that wrapped tool function returns correct dict."""
        tools_list = ["file-server.read-file"]
        
        result = ToolsWrapper(tools_list)
        tool = result[0]
        
        # Call the tool function
        output = tool.func(filename="test.txt")
        
        assert output["server"] == "file-server"
        assert output["tool"] == "read-file"
        assert "args" in output

    def test_tools_wrapper_tool_with_multiple_dots(self):
        """Test tool name with multiple dots."""
        tools_list = ["complex-server.sub.read-file"]
        
        result = ToolsWrapper(tools_list)
        
        assert len(result) == 1
        assert result[0].name == "sub.read-file"

class TestMakeExecutionAgent:
    """Test the make_execution_agent function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return Mock()

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return [
            Tool(name="tool1", description="Tool 1", func=lambda x: x),
            Tool(name="tool2", description="Tool 2", func=lambda x: x),
        ]

    def test_make_execution_agent_creation(self, mock_llm, mock_tools):
        """Test that agent is created successfully."""
        with patch("shardguard.core.execution_langchain.initialize_agent") as mock_init:
            mock_agent = Mock()
            mock_init.return_value = mock_agent
            
            result = make_execution_agent(mock_llm, mock_tools)
            
            assert result == mock_agent
            mock_init.assert_called_once()

    def test_make_execution_agent_parameters(self, mock_llm, mock_tools):
        """Test that agent is initialized with correct parameters."""
        with patch("shardguard.core.execution_langchain.initialize_agent") as mock_init:
            mock_agent = Mock()
            mock_init.return_value = mock_agent
            
            make_execution_agent(mock_llm, mock_tools)
            
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["llm"] == mock_llm
            assert len(call_kwargs["tools"]) == 2
            assert call_kwargs["agent"] == AgentType.ZERO_SHOT_REACT_DESCRIPTION
            assert call_kwargs["verbose"] is False
            assert call_kwargs["handle_parsing_errors"] is True

    def test_make_execution_agent_with_empty_tools(self, mock_llm):
        """Test agent creation with no tools."""
        with patch("shardguard.core.execution_langchain.initialize_agent") as mock_init:
            mock_agent = Mock()
            mock_init.return_value = mock_agent
            
            result = make_execution_agent(mock_llm, [])
            
            assert result == mock_agent
            call_kwargs = mock_init.call_args[1]
            assert len(call_kwargs["tools"]) == 0

    def test_make_execution_agent_prints_message(self, mock_llm, mock_tools, capsys):
        """Test that agent creation prints confirmation message."""
        with patch("shardguard.core.execution_langchain.initialize_agent") as mock_init:
            mock_agent = Mock()
            mock_init.return_value = mock_agent
            
            make_execution_agent(mock_llm, mock_tools)
            
            captured = capsys.readouterr()
            assert "[-] Using LangChain Agent" in captured.out

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_wrapper_with_tools_workflow(self):
        """Test complete workflow from wrapper to tools."""
        # Create mock GenericExecutionLLM
        mock_generic_llm = Mock()
        mock_generic_llm.provider_type = "ollama"
        mock_generic_llm.model = "llama3.2"
        mock_generic_llm.llm_provider = Mock()
        mock_generic_llm.llm_provider.generate_response = AsyncMock(
            return_value='{"action": "read", "file": "test.txt"}'
        )
        
        # Create wrapper
        wrapper = GenericExecutionLLMWrapper(mock_generic_llm)
        
        # Create tools
        tools_list = ["file-server.read-file", "file-server.write-file"]
        tools = ToolsWrapper(tools_list)
        
        # Test wrapper call
        response = await wrapper._acall("Read file test.txt")
        assert "action" in response
        
        # Verify tools were created
        assert len(tools) == 2
        assert tools[0].name == "read-file"

    def test_tools_to_agent_workflow(self):
        """Test workflow from tools creation to agent initialization."""

        tools_list = ["server.tool1", "server.tool2"]
        langchain_tools = ToolsWrapper(tools_list)
        
        # Create mock LLM
        mock_llm = Mock()
        
        with patch("shardguard.core.execution_langchain.initialize_agent") as mock_init:
            mock_agent = Mock()
            mock_init.return_value = mock_agent
            
            agent = make_execution_agent(mock_llm, langchain_tools)
            
            assert agent is not None
            assert mock_init.called
            call_kwargs = mock_init.call_args[1]
            assert len(call_kwargs["tools"]) == 2