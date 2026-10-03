// Gemini translation: a tool call id reused across turns keeps each turn's own
// result and name (upstream 9router #4273); `errorMessage` is stripped from tool
// schemas (#4283) while parameters that happen to share a keyword's name stay.
import { describe, it, expect } from "vitest";
import "./registerAll.js";
import { translateRequest, translateResponse, initState } from "../../open-sse/translator/index.js";
import { FORMATS } from "../../open-sse/translator/formats.js";
import { cleanJSONSchemaForAntigravity } from "../../open-sse/translator/formats/gemini.js";

const O2G = (body) => translateRequest(FORMATS.OPENAI, FORMATS.GEMINI, "m", body, true, null, "gemini");

const call = (id, name, args) => ({ id, type: "function", function: { name, arguments: JSON.stringify(args) } });

describe("OpenAI → Gemini with a reused tool call id", () => {
  const body = {
    messages: [
      { role: "user", content: "check both files" },
      { role: "assistant", content: null, tool_calls: [call("call_0", "read_file", { path: "a.txt" })] },
      { role: "tool", tool_call_id: "call_0", content: "contents of A" },
      { role: "assistant", content: null, tool_calls: [call("call_0", "list_dir", { path: "." })] },
      { role: "tool", tool_call_id: "call_0", content: "a.txt b.txt" },
      { role: "user", content: "thanks" },
    ],
  };

  it("pairs each result with its own turn and gives the second use a distinct id", () => {
    const out = O2G(body);
    const calls = out.contents.flatMap((c) => c.parts).filter((p) => p.functionCall).map((p) => p.functionCall);
    const results = out.contents.flatMap((c) => c.parts).filter((p) => p.functionResponse).map((p) => p.functionResponse);
    expect(calls.map((c) => [c.id, c.name])).toEqual([["call_0", "read_file"], ["call_0_2", "list_dir"]]);
    expect(results.map((r) => [r.id, r.name, r.response.result.result])).toEqual([
      ["call_0", "read_file", "contents of A"],
      ["call_0_2", "list_dir", "a.txt b.txt"],
    ]);
  });

  it("handles Claude models in Antigravity envelope with reused tool call id", () => {
    const out = translateRequest(FORMATS.OPENAI, FORMATS.ANTIGRAVITY, "claude-3-5-sonnet", body, true, null, "antigravity");
    const contents = out.request.contents;
    const calls = contents.flatMap((c) => c.parts).filter((p) => p.functionCall).map((p) => p.functionCall);
    const results = contents.flatMap((c) => c.parts).filter((p) => p.functionResponse).map((p) => p.functionResponse);
    expect(calls.map((c) => [c.id, c.name])).toEqual([["call_0", "read_file"], ["call_0_2", "list_dir"]]);
    expect(results.map((r) => [r.id, r.name, r.response.result])).toEqual([
      ["call_0", "read_file", "contents of A"],
      ["call_0_2", "list_dir", "a.txt b.txt"],
    ]);
  });

  it("leaves unique ids exactly as they were", () => {
    const out = O2G({
      messages: [
        { role: "user", content: "go" },
        { role: "assistant", content: null, tool_calls: [call("call_a", "read_file", { path: "a" }), call("call_b", "read_file", { path: "b" })] },
        { role: "tool", tool_call_id: "call_a", content: "A" },
        { role: "tool", tool_call_id: "call_b", content: "B" },
      ],
    });
    const results = out.contents.flatMap((c) => c.parts).filter((p) => p.functionResponse).map((p) => p.functionResponse);
    expect(results.map((r) => [r.id, r.response.result.result])).toEqual([["call_a", "A"], ["call_b", "B"]]);
  });
});

describe("cleanJSONSchemaForAntigravity", () => {
  it("strips errorMessage and other unsupported keywords", () => {
    const cleaned = cleanJSONSchemaForAntigravity({
      type: "object",
      properties: { path: { type: "string", minLength: 1, errorMessage: "path is required" } },
      required: ["path"],
      errorMessage: { required: "missing" },
    });
    expect(JSON.stringify(cleaned)).not.toContain("errorMessage");
    expect(JSON.stringify(cleaned)).not.toContain("minLength");
    expect(cleaned.properties.path.type).toBe("string");
  });

  it("keeps parameters whose names match a keyword", () => {
    const cleaned = cleanJSONSchemaForAntigravity({
      type: "object",
      properties: {
        title: { type: "string" },
        format: { type: "string", enum: ["md", "html"] },
        errorMessage: { type: "string" },
        default: { type: "boolean" },
      },
      required: ["title", "errorMessage"],
    });
    expect(Object.keys(cleaned.properties).sort()).toEqual(["default", "errorMessage", "format", "title"]);
    expect(cleaned.required).toEqual(["title", "errorMessage"]);
  });
});

describe("Gemini tool name mapping & restoration", () => {
  it("populates _toolNameMap when tools contain special characters", () => {
    const out = O2G({
      messages: [{ role: "user", content: "do it" }],
      tools: [
        {
          type: "function",
          function: {
            name: "mcp.browser/click",
            description: "click element",
            parameters: { type: "object", properties: { x: { type: "number" } } }
          }
        }
      ]
    });
    expect(out.tools[0].functionDeclarations[0].name).toBe("mcp.browser_click");
    expect(out._toolNameMap).toBeDefined();
    expect(out._toolNameMap.get("mcp.browser_click")).toBe("mcp.browser/click");
  });

  it("restores original tool name in response when toolNameMap is passed in state", () => {
    const toolNameMap = new Map([["mcp.browser_click", "mcp.browser/click"]]);
    const state = { ...initState(FORMATS.OPENAI), toolNameMap };
    const chunk = {
      candidates: [
        {
          content: {
            parts: [
              {
                functionCall: {
                  name: "mcp.browser_click",
                  args: { x: 100 }
                }
              }
            ]
          },
          finishReason: "STOP"
        }
      ]
    };
    const res = translateResponse(FORMATS.GEMINI, FORMATS.OPENAI, chunk, state);
    const tcChunk = res.find((r) => r.choices?.[0]?.delta?.tool_calls);
    expect(tcChunk).toBeDefined();
    expect(tcChunk.choices[0].delta.tool_calls[0].function.name).toBe("mcp.browser/click");
  });
});
