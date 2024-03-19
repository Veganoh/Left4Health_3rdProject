import { RequestContext, Text } from '@botonic/react'
import React from 'react'

export default class extends React.Component {
  static contextType = RequestContext

  render() {
    let context= this.context.getString
    return (
        <>
        <Text>Hi human! 👋</Text>
        <Text>
          This is for sure melanoma {context}
        </Text>
        <Text>Ask me something related! 😊</Text>
      </>
    )
  }
}