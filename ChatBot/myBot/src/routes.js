import React from 'react'
import { Text, Reply } from '@botonic/react'
import Melanoma from './actions/melanoma'

export const routes = [
  {
    path: 'initial',
    text: /hi/i,
    action: () => (
      <>
        <Text>Hello! Nice to meet you ;)</Text>
        <Text>
          How can I help you?
          <Reply payload='search'>Skin diagnostic</Reply>
          <Reply payload='track'>I need help</Reply>
        </Text>
      </>
    ),
  },
   { intent: 'Melanoma', action: Melanoma }
]